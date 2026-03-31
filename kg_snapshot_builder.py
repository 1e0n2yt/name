"""
kg_snapshot_builder.py
知识图谱时序快照构建模块（代码1）

功能：
    - 依据输入的微博话题以及 CSV 格式的文件进行知识图谱时序快照的构建与保存
    - 相同话题的知识图谱时序快照保存在同一个文件夹下，文件夹命名为 {话题名称}
    - 时序快照命名为生成的时间，格式为 {年}_{月}_{日}.pkl
    - 时序快照只保留最近 k 天（参数可配置，默认 7 天）
    - 知识图谱实体和边严格参照 social_media_models.py 中的数据模型

输入数据格式（CSV 列）：
    comment_id, user_id, post_id, content, created_at, likes, sentiment_score, keywords

输出数据格式：
    序列化的知识图谱对象（pkl 格式）
    包含节点：User、Post、Comment、Keyword、SentimentLabel
    包含边：User-Post关系、User-Comment关系、Comment-Sentiment关系、
           Comment-Keyword关系、Comment-时序关系

遵循 PEP8 规范，random_state=42 保证可复现性。
"""

import ast
import logging
import os
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from social_media_models import Comment, Keyword, Post, SentimentLabel, User

# 设置随机种子保证可复现性
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 节点类型常量
# ──────────────────────────────────────────────────────────────────────────────

class NodeType:
    """知识图谱节点类型常量"""
    USER = "user"
    POST = "post"
    COMMENT = "comment"
    KEYWORD = "keyword"
    SENTIMENT = "sentiment"


class EdgeType:
    """知识图谱边类型常量"""
    USER_WRITES_POST = "writes_post"          # 用户 → 发帖
    USER_WRITES_COMMENT = "writes_comment"    # 用户 → 评论
    COMMENT_BELONGS_TO_POST = "belongs_to"    # 评论 → 帖子
    COMMENT_HAS_SENTIMENT = "has_sentiment"   # 评论 → 情感标签
    COMMENT_CONTAINS_KEYWORD = "contains"     # 评论 → 关键词
    COMMENT_FOLLOWS = "follows"               # 评论1 → 评论2（时序关系）


# ──────────────────────────────────────────────────────────────────────────────
# 知识图谱快照类
# ──────────────────────────────────────────────────────────────────────────────

class KGSnapshot:
    """
    单个时序快照对象

    封装某一天的知识图谱及其统计信息。

    Attributes:
        date (datetime): 快照日期
        graph (nx.DiGraph): 有向知识图谱
        sentiment_distribution (Dict[str, float]): 当日情感分布
        comment_count (int): 当日评论数
        high_risk_user_ids (List[str]): 高风险用户 ID 列表
    """

    def __init__(self, date: datetime, graph: nx.DiGraph):
        self.date = date
        self.graph = graph
        self.sentiment_distribution: Dict[str, float] = {
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 0.0,
        }
        self.comment_count: int = 0
        self.high_risk_user_ids: List[str] = []
        self._compute_statistics()

    def _compute_statistics(self) -> None:
        """从图中计算统计信息"""
        comment_nodes = [
            (nid, data)
            for nid, data in self.graph.nodes(data=True)
            if data.get("node_type") == NodeType.COMMENT
        ]
        self.comment_count = len(comment_nodes)

        if self.comment_count == 0:
            return

        pos_count, neu_count, neg_count = 0, 0, 0
        high_risk_users = set()

        for nid, data in comment_nodes:
            label = data.get("sentiment_label", "neutral")
            if label == SentimentLabel.POSITIVE:
                pos_count += 1
            elif label == SentimentLabel.NEGATIVE:
                neg_count += 1
            else:
                neu_count += 1

            # 判断高风险用户（情感分数 <= -0.6）
            if data.get("sentiment_score", 0.0) <= -0.6:
                user_id = data.get("user_id", "")
                if user_id:
                    high_risk_users.add(user_id)

        total = self.comment_count
        self.sentiment_distribution = {
            "positive": pos_count / total,
            "neutral": neu_count / total,
            "negative": neg_count / total,
        }
        self.high_risk_user_ids = list(high_risk_users)

    def get_feature_vector(self) -> List[float]:
        """
        返回快照特征向量，用于 KGCN 模型输入

        Returns:
            List[float]: 特征向量 [pos_ratio, neu_ratio, neg_ratio, comment_count_normalized,
                          high_risk_user_ratio]
        """
        dist = self.sentiment_distribution
        comment_norm = min(1.0, self.comment_count / 1000.0)
        total_users = len(set(
            data.get("user_id", "")
            for _, data in self.graph.nodes(data=True)
            if data.get("node_type") == NodeType.COMMENT
        ))
        high_risk_ratio = (
            len(self.high_risk_user_ids) / max(1, total_users)
        )
        return [
            dist["positive"],
            dist["neutral"],
            dist["negative"],
            comment_norm,
            high_risk_ratio,
        ]

    def __repr__(self) -> str:
        return (
            f"KGSnapshot(date={self.date.strftime('%Y-%m-%d')}, "
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"comments={self.comment_count}, "
            f"neg_ratio={self.sentiment_distribution['negative']:.2f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 知识图谱构建器
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """
    知识图谱构建器

    从评论数据（Comment 对象列表）构建当天的有向知识图谱，
    包含 User、Post、Comment、Keyword、SentimentLabel 节点及各类关系边。
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def build(
        self,
        comments: List[Comment],
        users: Optional[Dict[str, User]] = None,
        posts: Optional[Dict[str, Post]] = None,
    ) -> nx.DiGraph:
        """
        构建知识图谱

        Args:
            comments (List[Comment]): 当天的评论列表
            users (Optional[Dict[str, User]]): 用户字典，key 为 user_id；
                                               为 None 时自动构建占位用户节点
            posts (Optional[Dict[str, Post]]): 帖子字典，key 为 post_id；
                                               为 None 时自动构建占位帖子节点

        Returns:
            nx.DiGraph: 构建完成的有向知识图谱
        """
        self.graph = nx.DiGraph()

        # 添加情感标签节点（固定 3 个）
        self._add_sentiment_nodes()

        # 添加评论及关联节点
        for comment in comments:
            self._add_comment_node(comment)
            self._add_or_get_user_node(comment.user_id, users)
            self._add_or_get_post_node(comment.post_id, posts)
            self._add_keyword_nodes(comment)

            # 添加关系边
            self._add_user_comment_edge(comment)
            self._add_comment_post_edge(comment)
            self._add_comment_sentiment_edge(comment)
            self._add_comment_keyword_edges(comment)

        # 添加时序关系（按评论时间排序后两两连接）
        self._add_temporal_edges(comments)

        logger.debug(
            "图构建完成：%d 节点，%d 条边",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法：节点添加
    # ──────────────────────────────────────────────────────────────────────────

    def _add_sentiment_nodes(self) -> None:
        """添加三种情感标签节点"""
        for label in [SentimentLabel.POSITIVE, SentimentLabel.NEUTRAL, SentimentLabel.NEGATIVE]:
            node_id = f"sentiment_{label}"
            self.graph.add_node(node_id, node_type=NodeType.SENTIMENT, label=label)

    def _add_comment_node(self, comment: Comment) -> None:
        """添加评论节点"""
        node_id = f"comment_{comment.comment_id}"
        sentiment_label = comment.get_sentiment_label()
        self.graph.add_node(
            node_id,
            node_type=NodeType.COMMENT,
            comment_id=comment.comment_id,
            user_id=comment.user_id,
            post_id=comment.post_id,
            content_clean=comment.content_clean,
            created_at=comment.created_at,
            likes=comment.likes,
            replies=comment.replies,
            sentiment_score=comment.sentiment_score,
            sentiment_label=sentiment_label,
            keywords=comment.keywords,
        )

    def _add_or_get_user_node(
        self, user_id: str, users: Optional[Dict[str, User]]
    ) -> None:
        """添加用户节点（若不存在则创建占位节点）"""
        node_id = f"user_{user_id}"
        if node_id in self.graph:
            return

        if users and user_id in users:
            u = users[user_id]
            self.graph.add_node(
                node_id,
                node_type=NodeType.USER,
                user_id=u.user_id,
                created_at=u.created_at,
                followers=u.followers,
                following=u.following,
                verified=u.verified,
                total_posts=u.total_posts,
                engagement_rate=u.engagement_rate,
            )
        else:
            # 占位用户节点（数据不全时）
            self.graph.add_node(
                node_id,
                node_type=NodeType.USER,
                user_id=user_id,
                created_at=datetime.now(),
                followers=0,
                following=0,
                verified=False,
                total_posts=0,
                engagement_rate=0.0,
            )

    def _add_or_get_post_node(
        self, post_id: str, posts: Optional[Dict[str, Post]]
    ) -> None:
        """添加帖子节点（若不存在则创建占位节点）"""
        node_id = f"post_{post_id}"
        if node_id in self.graph:
            return

        if posts and post_id in posts:
            p = posts[post_id]
            self.graph.add_node(
                node_id,
                node_type=NodeType.POST,
                post_id=p.post_id,
                user_id=p.user_id,
                content_clean=p.content_clean,
                created_at=p.created_at,
                likes=p.likes,
                replies=p.replies,
                reposts=p.reposts,
                viral_coefficient=p.viral_coefficient,
                sentiment_score=p.sentiment_score,
                keywords=p.keywords,
            )
        else:
            self.graph.add_node(
                node_id,
                node_type=NodeType.POST,
                post_id=post_id,
                user_id="",
                content_clean="",
                created_at=datetime.now(),
                likes=0,
                replies=0,
                reposts=0,
                viral_coefficient=0.0,
                sentiment_score=0.0,
                keywords=[],
            )

    def _add_keyword_nodes(self, comment: Comment) -> None:
        """为评论中的每个关键词添加关键词节点"""
        for kw in comment.keywords:
            node_id = f"keyword_{kw}"
            if node_id not in self.graph:
                self.graph.add_node(
                    node_id,
                    node_type=NodeType.KEYWORD,
                    keyword=kw,
                    frequency=1,
                    tf_idf=0.0,
                    positive_context_count=0,
                    negative_context_count=0,
                )
            else:
                # 累加频次
                self.graph.nodes[node_id]["frequency"] += 1
                if comment.sentiment_score > 0.1:
                    self.graph.nodes[node_id]["positive_context_count"] += 1
                elif comment.sentiment_score < -0.1:
                    self.graph.nodes[node_id]["negative_context_count"] += 1

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法：边添加
    # ──────────────────────────────────────────────────────────────────────────

    def _add_user_comment_edge(self, comment: Comment) -> None:
        """添加 用户→评论 边（用户编写评论关系）"""
        self.graph.add_edge(
            f"user_{comment.user_id}",
            f"comment_{comment.comment_id}",
            edge_type=EdgeType.USER_WRITES_COMMENT,
            weight=1.0,
        )

    def _add_comment_post_edge(self, comment: Comment) -> None:
        """添加 评论→帖子 边（评论归属帖子关系）"""
        self.graph.add_edge(
            f"comment_{comment.comment_id}",
            f"post_{comment.post_id}",
            edge_type=EdgeType.COMMENT_BELONGS_TO_POST,
            weight=1.0,
        )

    def _add_comment_sentiment_edge(self, comment: Comment) -> None:
        """添加 评论→情感标签 边（情感关联关系），边权为情感分数绝对值"""
        sentiment_label = comment.get_sentiment_label()
        self.graph.add_edge(
            f"comment_{comment.comment_id}",
            f"sentiment_{sentiment_label}",
            edge_type=EdgeType.COMMENT_HAS_SENTIMENT,
            weight=abs(comment.sentiment_score),
        )

    def _add_comment_keyword_edges(self, comment: Comment) -> None:
        """添加 评论→关键词 边（关键词包含关系），边权为 TF-IDF 值（简化使用均值）"""
        for kw in comment.keywords:
            self.graph.add_edge(
                f"comment_{comment.comment_id}",
                f"keyword_{kw}",
                edge_type=EdgeType.COMMENT_CONTAINS_KEYWORD,
                weight=1.0,
            )

    def _add_temporal_edges(self, comments: List[Comment]) -> None:
        """
        添加时序关系边（评论1 → 评论2，按时间顺序）

        时序关系是 KGCN 捕捉情感演变的核心：
        - 按评论发布时间排序
        - 相邻评论之间添加 follows 边
        - 边权为情感分数的变化量（绝对值），变化越大权重越高
        """
        sorted_comments = sorted(comments, key=lambda c: c.created_at)
        for i in range(len(sorted_comments) - 1):
            c_curr = sorted_comments[i]
            c_next = sorted_comments[i + 1]
            sentiment_delta = abs(c_next.sentiment_score - c_curr.sentiment_score)
            self.graph.add_edge(
                f"comment_{c_curr.comment_id}",
                f"comment_{c_next.comment_id}",
                edge_type=EdgeType.COMMENT_FOLLOWS,
                weight=float(sentiment_delta),
                time_delta=(c_next.created_at - c_curr.created_at).total_seconds(),
            )


# ──────────────────────────────────────────────────────────────────────────────
# CSV 数据加载器
# ──────────────────────────────────────────────────────────────────────────────

class CSVDataLoader:
    """
    CSV 数据加载器

    将 CSV 文件解析为 Comment 对象列表，支持 created_at 日期过滤。

    CSV 列格式：
        comment_id, user_id, post_id, content, created_at, likes,
        sentiment_score, keywords
    """

    REQUIRED_COLUMNS = {
        "comment_id", "user_id", "post_id", "content",
        "created_at", "likes", "sentiment_score", "keywords",
    }

    def load(self, csv_path: str) -> pd.DataFrame:
        """
        加载 CSV 文件

        Args:
            csv_path (str): CSV 文件路径

        Returns:
            pd.DataFrame: 加载并预处理后的数据框

        Raises:
            FileNotFoundError: CSV 文件不存在
            ValueError: CSV 缺少必要列
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

        df = pd.read_csv(csv_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV 文件缺少必要列：{missing}")

        # 解析时间列
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.dropna(subset=["created_at"])

        # 解析关键词列（支持字符串列表格式）
        df["keywords"] = df["keywords"].apply(self._parse_keywords)

        # 情感分数截断到 [-1, 1]
        df["sentiment_score"] = df["sentiment_score"].clip(-1.0, 1.0)

        # 点赞数取非负整数
        df["likes"] = df["likes"].fillna(0).astype(int).clip(lower=0)

        logger.info("加载 CSV 完成：共 %d 条记录，文件：%s", len(df), csv_path)
        return df

    @staticmethod
    def _parse_keywords(raw) -> List[str]:
        """解析关键词字段（支持多种格式）"""
        if isinstance(raw, list):
            return [str(k) for k in raw]
        if pd.isna(raw) or raw == "":
            return []
        if isinstance(raw, str):
            raw = raw.strip()
            # 尝试解析 Python list 字符串格式
            if raw.startswith("["):
                try:
                    parsed = ast.literal_eval(raw)
                    return [str(k) for k in parsed]
                except (ValueError, SyntaxError):
                    pass
            # 逗号分隔
            return [k.strip() for k in raw.split(",") if k.strip()]
        return []

    def to_comments(self, df: pd.DataFrame) -> List[Comment]:
        """
        将数据框转换为 Comment 对象列表

        Args:
            df (pd.DataFrame): 已加载的数据框

        Returns:
            List[Comment]: Comment 对象列表
        """
        comments = []
        for _, row in df.iterrows():
            try:
                comment = Comment(
                    comment_id=str(row["comment_id"]),
                    user_id=str(row["user_id"]),
                    post_id=str(row["post_id"]),
                    content_clean=str(row.get("content", "")),
                    created_at=row["created_at"].to_pydatetime(),
                    likes=int(row["likes"]),
                    replies=int(row.get("replies", 0)),
                    sentiment_score=float(row["sentiment_score"]),
                    keywords=row["keywords"],
                )
                comments.append(comment)
            except (ValueError, KeyError) as exc:
                logger.warning("跳过无效评论行：%s，错误：%s", row.get("comment_id"), exc)
        return comments


# ──────────────────────────────────────────────────────────────────────────────
# 时序快照管理器（主入口）
# ──────────────────────────────────────────────────────────────────────────────

class TemporalSnapshotManager:
    """
    知识图谱时序快照管理器

    负责：
    1. 从 CSV 文件按日期分组构建每日知识图谱快照
    2. 将快照序列化保存到本地（pkl 格式）
    3. 自动清理超过 keep_days 天的旧快照
    4. 支持加载已保存的快照

    目录结构：
        {snapshot_dir}/{topic_name}/{YYYY_MM_DD}.pkl
    """

    def __init__(
        self,
        snapshot_dir: str = "./snapshots",
        keep_days: int = 7,
    ):
        """
        初始化快照管理器

        Args:
            snapshot_dir (str): 快照根目录，默认 "./snapshots"
            keep_days (int): 保留最近天数，默认 7 天
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.keep_days = keep_days
        self.loader = CSVDataLoader()
        self.builder = KnowledgeGraphBuilder()

    def build_and_save(
        self,
        csv_path: str,
        topic_name: str,
        users: Optional[Dict[str, User]] = None,
        posts: Optional[Dict[str, Post]] = None,
    ) -> List[KGSnapshot]:
        """
        从 CSV 文件构建知识图谱时序快照并保存到本地

        Args:
            csv_path (str): 输入 CSV 文件路径
            topic_name (str): 话题名称（用于目录命名）
            users (Optional[Dict[str, User]]): 用户信息字典，可选
            posts (Optional[Dict[str, Post]]): 帖子信息字典，可选

        Returns:
            List[KGSnapshot]: 构建并保存的快照列表（按日期升序）
        """
        # 1. 加载数据
        df = self.loader.load(csv_path)
        all_comments = self.loader.to_comments(df)
        logger.info("话题「%s」：加载到 %d 条评论", topic_name, len(all_comments))

        # 2. 计算时间窗口（保留最近 keep_days 天）
        today = datetime.now().date()
        cutoff_date = today - timedelta(days=self.keep_days - 1)

        # 3. 按日期分组评论
        comments_by_date: Dict[str, List[Comment]] = {}
        for comment in all_comments:
            day = comment.created_at.date()
            if day >= cutoff_date:
                key = day.strftime("%Y_%m_%d")
                comments_by_date.setdefault(key, []).append(comment)

        if not comments_by_date:
            logger.warning("话题「%s」：在最近 %d 天内无评论数据", topic_name, self.keep_days)
            return []

        # 4. 确保目录存在
        topic_dir = self.snapshot_dir / topic_name
        topic_dir.mkdir(parents=True, exist_ok=True)

        # 5. 构建并保存各日快照
        snapshots = []
        for date_key in sorted(comments_by_date.keys()):
            day_comments = comments_by_date[date_key]
            snap_date = datetime.strptime(date_key, "%Y_%m_%d")

            logger.info(
                "话题「%s」：构建 %s 快照（%d 条评论）",
                topic_name, date_key, len(day_comments),
            )

            graph = self.builder.build(day_comments, users=users, posts=posts)
            snapshot = KGSnapshot(date=snap_date, graph=graph)

            # 保存到文件
            save_path = topic_dir / f"{date_key}.pkl"
            self._save_snapshot(snapshot, save_path)
            snapshots.append(snapshot)

            logger.info(
                "快照已保存：%s（neg_ratio=%.2f）",
                save_path,
                snapshot.sentiment_distribution["negative"],
            )

        # 6. 清理过期快照
        self._cleanup_old_snapshots(topic_dir)

        return snapshots

    def load_snapshots(self, topic_name: str) -> List[KGSnapshot]:
        """
        加载指定话题的所有已保存快照（按日期升序）

        Args:
            topic_name (str): 话题名称

        Returns:
            List[KGSnapshot]: 快照列表

        Raises:
            FileNotFoundError: 话题目录不存在
        """
        topic_dir = self.snapshot_dir / topic_name
        if not topic_dir.exists():
            raise FileNotFoundError(
                f"话题目录不存在：{topic_dir}\n"
                f"请先调用 build_and_save() 构建快照。"
            )

        pkl_files = sorted(topic_dir.glob("*.pkl"))
        snapshots = []
        for pkl_path in pkl_files:
            try:
                snap = self._load_snapshot(pkl_path)
                snapshots.append(snap)
                logger.debug("加载快照：%s", pkl_path.name)
            except Exception as exc:
                logger.error("加载快照失败：%s，错误：%s", pkl_path, exc)

        logger.info("话题「%s」：加载了 %d 个快照", topic_name, len(snapshots))
        return snapshots

    def list_topics(self) -> List[str]:
        """
        列出所有已保存快照的话题名称

        Returns:
            List[str]: 话题名称列表
        """
        if not self.snapshot_dir.exists():
            return []
        return [d.name for d in self.snapshot_dir.iterdir() if d.is_dir()]

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_snapshot(snapshot: KGSnapshot, path: Path) -> None:
        """序列化快照到 pkl 文件"""
        with open(path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_snapshot(path: Path) -> KGSnapshot:
        """从 pkl 文件反序列化快照"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _cleanup_old_snapshots(self, topic_dir: Path) -> None:
        """
        删除超过 keep_days 天的旧快照文件

        保留最新的 keep_days 个快照（按文件名日期排序）。
        """
        pkl_files = sorted(topic_dir.glob("*.pkl"))
        files_to_delete = pkl_files[: max(0, len(pkl_files) - self.keep_days)]
        for f in files_to_delete:
            f.unlink()
            logger.info("清理过期快照：%s", f.name)


# ──────────────────────────────────────────────────────────────────────────────
# 便捷函数
# ──────────────────────────────────────────────────────────────────────────────

def build_snapshots(
    csv_path: str,
    topic_name: str,
    snapshot_dir: str = "./snapshots",
    keep_days: int = 7,
    users: Optional[Dict[str, User]] = None,
    posts: Optional[Dict[str, Post]] = None,
) -> List[KGSnapshot]:
    """
    便捷函数：构建并保存知识图谱时序快照

    Args:
        csv_path (str): 输入 CSV 文件路径
        topic_name (str): 话题名称
        snapshot_dir (str): 快照保存根目录，默认 "./snapshots"
        keep_days (int): 保留最近天数，默认 7
        users (Optional[Dict[str, User]]): 用户信息字典，可选
        posts (Optional[Dict[str, Post]]): 帖子信息字典，可选

    Returns:
        List[KGSnapshot]: 构建完成的快照列表

    Example:
        >>> snapshots = build_snapshots(
        ...     csv_path="data.csv",
        ...     topic_name="某热点话题",
        ...     snapshot_dir="./snapshots",
        ...     keep_days=7,
        ... )
        >>> for snap in snapshots:
        ...     print(snap)
    """
    manager = TemporalSnapshotManager(
        snapshot_dir=snapshot_dir,
        keep_days=keep_days,
    )
    return manager.build_and_save(
        csv_path=csv_path,
        topic_name=topic_name,
        users=users,
        posts=posts,
    )


def load_snapshots(
    topic_name: str,
    snapshot_dir: str = "./snapshots",
) -> List[KGSnapshot]:
    """
    便捷函数：加载指定话题的已保存快照

    Args:
        topic_name (str): 话题名称
        snapshot_dir (str): 快照根目录，默认 "./snapshots"

    Returns:
        List[KGSnapshot]: 快照列表（按日期升序）
    """
    manager = TemporalSnapshotManager(snapshot_dir=snapshot_dir)
    return manager.load_snapshots(topic_name=topic_name)


# ──────────────────────────────────────────────────────────────────────────────
# 可运行示例
# ──────────────────────────────────────────────────────────────────────────────

def _generate_demo_csv(csv_path: str, num_days: int = 7, comments_per_day: int = 20) -> None:
    """生成演示用 CSV 数据文件"""
    import random as _random
    _random.seed(42)
    np.random.seed(42)

    base_date = datetime.now() - timedelta(days=num_days - 1)
    rows = []
    comment_id = 1

    for day in range(num_days):
        day_date = base_date + timedelta(days=day)
        # 模拟情感随时间恶化的趋势
        neg_bias = -0.1 * day  # 每天负面情感增加

        n = comments_per_day + _random.randint(-5, 10)
        for _ in range(n):
            hour = _random.randint(8, 22)
            minute = _random.randint(0, 59)
            ts = day_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

            sentiment = float(np.clip(
                np.random.normal(0.0 + neg_bias, 0.4), -1.0, 1.0
            ))
            user_id = f"u{_random.randint(1, 50):03d}"
            post_id = f"p{_random.randint(1, 5):03d}"
            kws = _random.sample(
                ["质量", "服务", "价格", "失望", "好用", "推荐", "差劲", "退款"], 2
            )
            rows.append({
                "comment_id": f"c{comment_id:05d}",
                "user_id": user_id,
                "post_id": post_id,
                "content": f"评论内容示例 {comment_id}",
                "created_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "likes": _random.randint(0, 100),
                "sentiment_score": round(sentiment, 4),
                "keywords": str(kws),
            })
            comment_id += 1

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("演示 CSV 已生成：%s（共 %d 条）", csv_path, len(rows))


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("kg_snapshot_builder.py 知识图谱时序快照构建示例")
    print("=" * 60)

    # 使用临时目录作为演示
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = os.path.join(tmp_dir, "demo_comments.csv")
        snap_dir = os.path.join(tmp_dir, "snapshots")
        topic = "某热点话题"

        # 1. 生成演示 CSV
        _generate_demo_csv(csv_file, num_days=7, comments_per_day=20)

        # 2. 构建并保存快照
        print(f"\n[步骤1] 构建知识图谱时序快照...")
        snapshots = build_snapshots(
            csv_path=csv_file,
            topic_name=topic,
            snapshot_dir=snap_dir,
            keep_days=7,
        )
        print(f"  成功构建 {len(snapshots)} 个快照")

        # 3. 显示快照信息
        print(f"\n[步骤2] 快照详情：")
        for snap in snapshots:
            dist = snap.sentiment_distribution
            print(
                f"  {snap.date.strftime('%Y-%m-%d')}  "
                f"评论数={snap.comment_count}  "
                f"正={dist['positive']:.2f} 中={dist['neutral']:.2f} 负={dist['negative']:.2f}  "
                f"高危用户={len(snap.high_risk_user_ids)}"
            )

        # 4. 重新加载快照
        print(f"\n[步骤3] 从磁盘加载快照...")
        loaded = load_snapshots(topic, snapshot_dir=snap_dir)
        print(f"  成功加载 {len(loaded)} 个快照")

        # 5. 验证图结构
        if loaded:
            sample = loaded[-1]
            print(f"\n[步骤4] 最新快照图结构验证：")
            print(f"  节点数: {sample.graph.number_of_nodes()}")
            print(f"  边数:   {sample.graph.number_of_edges()}")
            node_types = {}
            for _, d in sample.graph.nodes(data=True):
                t = d.get("node_type", "unknown")
                node_types[t] = node_types.get(t, 0) + 1
            for nt, cnt in node_types.items():
                print(f"    {nt}: {cnt} 个节点")
            print(f"  特征向量: {sample.get_feature_vector()}")

        print("\n✅ 知识图谱时序快照构建验证通过")
