"""
phase1_kg_snapshot/snapshot.py - 时序图快照生成与管理

设计说明（对应问题2）：
  - TimeSlot 节点聚合时间信息（小时/天粒度）
  - 每个快照对应一个时间窗口的子图 + 聚合统计
  - 快照序列是 KGCN 时序建模的输入
  - 支持按天生成快照，保存为 pickle 文件，供 Phase 2 读取
"""

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import networkx as nx

from data_models import NodeType, SentimentRange


@dataclass
class GraphSnapshot:
    """
    单个时序图快照

    包含：
      - snapshot_id  : 唯一标识（通常为 "topic_{id}_{date}"）
      - topic_id     : 监控话题 ID
      - start_time   : 快照时间窗口起始
      - end_time     : 快照时间窗口结束
      - graph        : 该窗口内的子图（nx.DiGraph）
      - 聚合统计属性  : 情感分布、节点/边计数等，供 KGCN 特征提取使用
    """

    snapshot_id: str
    topic_id: str
    start_time: datetime
    end_time: datetime
    graph: Any = field(default=None, repr=False)  # nx.DiGraph

    # 聚合统计（由 _compute_stats() 填充）
    num_nodes: int = 0
    num_edges: int = 0
    num_users: int = 0
    num_comments: int = 0
    num_posts: int = 0

    # 情感统计
    avg_sentiment: float = 0.5          # [0, 1]
    negative_ratio: float = 0.0         # [0, 1]
    extreme_negative_ratio: float = 0.0  # [0, 1]
    positive_ratio: float = 0.0         # [0, 1]
    sentiment_std: float = 0.0          # 情感分数标准差（越大越极化）

    # 活跃度统计
    active_users: int = 0               # 本窗口内发言的用户数
    new_users: int = 0                  # 相对于上一快照新增用户数

    # 传播统计
    avg_virality: float = 0.0           # 博文平均传播分数
    total_likes: int = 0
    total_reposts: int = 0

    # 异常信号
    sentiment_delta: float = 0.0        # 与上一快照的平均情感差值
    has_anomaly: bool = False           # 是否检测到异常信号

    def compute_stats(self, prev_snapshot: Optional["GraphSnapshot"] = None):
        """
        从图数据计算聚合统计属性。
        可传入上一个快照用于计算增量指标（sentiment_delta / new_users）。
        """
        if self.graph is None:
            return

        g: nx.DiGraph = self.graph
        self.num_nodes = g.number_of_nodes()
        self.num_edges = g.number_of_edges()

        users, comments, posts = [], [], []
        sentiment_scores = []

        for nid, data in g.nodes(data=True):
            nt = data.get("node_type", "")
            if nt == NodeType.USER:
                users.append(nid)
            elif nt == NodeType.COMMENT:
                comments.append(nid)
                s = data.get("sentiment_score")
                if s is not None:
                    sentiment_scores.append(float(s))
            elif nt == NodeType.POST:
                posts.append(nid)
                s = data.get("sentiment_score")
                if s is not None:
                    sentiment_scores.append(float(s))

        self.num_users = len(users)
        self.num_comments = len(comments)
        self.num_posts = len(posts)
        self.active_users = self.num_users

        if sentiment_scores:
            n = len(sentiment_scores)
            self.avg_sentiment = round(sum(sentiment_scores) / n, 4)
            self.negative_ratio = round(
                sum(1 for s in sentiment_scores if SentimentRange.is_negative(s)) / n, 4
            )
            self.extreme_negative_ratio = round(
                sum(
                    1
                    for s in sentiment_scores
                    if s < SentimentRange.EXTREME_NEGATIVE_MAX
                )
                / n,
                4,
            )
            self.positive_ratio = round(
                sum(
                    1
                    for s in sentiment_scores
                    if s >= SentimentRange.NEUTRAL_MAX
                )
                / n,
                4,
            )
            mean = self.avg_sentiment
            variance = sum((s - mean) ** 2 for s in sentiment_scores) / n
            self.sentiment_std = round(variance ** 0.5, 4)

        # 传播统计
        total_likes = sum(
            g.nodes[nid].get("likes_count", 0) for nid in posts
        )
        total_reposts = sum(
            g.nodes[nid].get("reposts_count", 0) for nid in posts
        )
        virality_scores = [
            g.nodes[nid].get("virality_score", 0.0) for nid in posts
        ]
        self.total_likes = total_likes
        self.total_reposts = total_reposts
        self.avg_virality = (
            round(sum(virality_scores) / len(virality_scores), 4)
            if virality_scores
            else 0.0
        )

        # 增量指标（需要上一快照）
        if prev_snapshot is not None:
            self.sentiment_delta = round(
                self.avg_sentiment - prev_snapshot.avg_sentiment, 4
            )
            prev_user_ids = {
                data.get("user_id")
                for _, data in prev_snapshot.graph.nodes(data=True)
                if data.get("node_type") == NodeType.USER
            }
            curr_user_ids = {
                g.nodes[nid].get("user_id") for nid in users
            }
            self.new_users = len(curr_user_ids - prev_user_ids)

        # 异常信号检测（简单规则）
        self.has_anomaly = (
            self.extreme_negative_ratio > 0.3
            or (self.sentiment_delta < -0.1 and prev_snapshot is not None)
            or (self.new_users > self.num_users * 0.3 and prev_snapshot is not None)
        )


class TemporalSnapshotGenerator:
    """
    时序快照生成器

    工作流：
      1. 接收完整知识图谱 + 评论时间戳信息
      2. 按时间窗口（默认每天）切割子图
      3. 计算每个快照的统计属性
      4. 返回快照序列（List[GraphSnapshot]）

    用途：
      - Phase 1 调用 generate() 生成并保存快照
      - Phase 2 加载快照序列，提取时序特征供 KGCN 使用
    """

    def __init__(self, granularity: str = "day"):
        """
        参数：
          granularity: "hour" | "day" | "week"
        """
        self.granularity = granularity

    def _get_slot_boundaries(
        self, start: datetime, end: datetime
    ) -> List[tuple]:
        """
        根据粒度生成时间槽列表 [(slot_start, slot_end), ...]。
        """
        slots = []
        delta = {
            "hour": timedelta(hours=1),
            "day": timedelta(days=1),
            "week": timedelta(weeks=1),
        }[self.granularity]

        current = datetime(start.year, start.month, start.day)
        if self.granularity == "hour":
            current = datetime(start.year, start.month, start.day, start.hour)

        while current <= end:
            slots.append((current, current + delta))
            current += delta
        return slots

    def generate(
        self,
        full_graph: nx.DiGraph,
        topic_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[GraphSnapshot]:
        """
        从完整知识图谱中按时间窗口切割，生成时序快照序列。

        参数：
          full_graph  : 完整的知识图谱（nx.DiGraph）
          topic_id    : 话题 ID
          start_time  : 数据起始时间
          end_time    : 数据结束时间

        返回：
          按时间升序排列的 GraphSnapshot 列表
        """
        slots = self._get_slot_boundaries(start_time, end_time)
        snapshots: List[GraphSnapshot] = []
        prev_snapshot: Optional[GraphSnapshot] = None

        for slot_start, slot_end in slots:
            # 筛选该时间窗口内的节点
            nodes_in_slot = []
            for nid, data in full_graph.nodes(data=True):
                ts = data.get("created_at") or data.get("start_time")
                if ts is not None and slot_start <= ts < slot_end:
                    nodes_in_slot.append(nid)

            if not nodes_in_slot:
                continue

            # 构建子图（包含这些节点以及它们之间的边）
            subgraph = full_graph.subgraph(nodes_in_slot).copy()

            # 生成快照 ID
            date_str = slot_start.strftime("%Y%m%d_%H" if self.granularity == "hour" else "%Y%m%d")
            snap_id = f"{topic_id}_{date_str}"

            snapshot = GraphSnapshot(
                snapshot_id=snap_id,
                topic_id=topic_id,
                start_time=slot_start,
                end_time=slot_end,
                graph=subgraph,
            )
            snapshot.compute_stats(prev_snapshot=prev_snapshot)
            snapshots.append(snapshot)
            prev_snapshot = snapshot

        return snapshots

    @staticmethod
    def get_recent_snapshots(
        snapshots: List[GraphSnapshot], n_days: int = 3
    ) -> List[GraphSnapshot]:
        """
        获取最近 n_days 天的快照（供 KGCN 使用）。

        Phase 2 中调用此方法获取近3日快照：
          recent = TemporalSnapshotGenerator.get_recent_snapshots(snapshots, n_days=3)
        """
        if not snapshots:
            return []
        sorted_snaps = sorted(snapshots, key=lambda s: s.start_time)
        return sorted_snaps[-n_days:]

    @staticmethod
    def extract_temporal_features(
        snapshots: List[GraphSnapshot],
    ) -> List[Dict]:
        """
        从快照序列中提取时序特征向量，供 KGCN 模型输入。

        每个快照对应一个特征字典，包含：
          - avg_sentiment        : 平均情感分数
          - negative_ratio       : 负面占比
          - extreme_negative_ratio: 极端负面占比
          - sentiment_std        : 情感标准差（极化程度）
          - sentiment_delta      : 与前一快照的情感差值
          - active_users         : 活跃用户数（归一化）
          - new_users            : 新增用户数（归一化）
          - avg_virality         : 平均传播分数
          - has_anomaly          : 是否有异常信号（0/1）
        """
        features = []
        for snap in snapshots:
            features.append(
                {
                    "snapshot_id": snap.snapshot_id,
                    "start_time": snap.start_time.isoformat(),
                    "avg_sentiment": snap.avg_sentiment,
                    "negative_ratio": snap.negative_ratio,
                    "extreme_negative_ratio": snap.extreme_negative_ratio,
                    "positive_ratio": snap.positive_ratio,
                    "sentiment_std": snap.sentiment_std,
                    "sentiment_delta": snap.sentiment_delta,
                    "active_users": snap.active_users,
                    "new_users": snap.new_users,
                    "num_comments": snap.num_comments,
                    "avg_virality": snap.avg_virality,
                    "has_anomaly": int(snap.has_anomaly),
                }
            )
        return features
