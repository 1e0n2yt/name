"""
phase2_prediction/opinion_level.py - 多维舆情等级评分

评分维度（4维加权）：
  1. 情感维度   (40%) : 负面占比 + 极端负面
  2. 传播维度   (25%) : 评论数、用户数、传播速度
  3. 影响力维度 (20%) : 高粉用户、认证用户、KOL 参与
  4. 趋势维度   (15%) : 情感变化斜率

等级映射（1-5）：
  L1 ( 0-25)  : 绿色 - 正常
  L2 (25-45)  : 蓝色 - 轻微关注
  L3 (45-62)  : 黄色 - 需要监控
  L4 (62-78)  : 橙色 - 高度警惕
  L5 (78-100) : 红色 - 紧急预警
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from data_models import NodeType, SentimentRange
from phase1_kg_snapshot.snapshot import GraphSnapshot
from phase2_prediction.kgcn_model import PredictionResult


@dataclass
class OpinionScore:
    """舆情综合评分结果。"""
    snapshot_id: str

    # 各维度分数 [0, 100]
    sentiment_score: float = 0.0
    propagation_score: float = 0.0
    influence_score: float = 0.0
    trend_score: float = 0.0

    # 综合分数与等级
    final_score: float = 0.0
    level: int = 1        # 1-5
    level_label: str = "正常"  # 绿/蓝/黄/橙/红
    level_color: str = "green"

    # KGCN 预测修正后的等级
    adjusted_level: int = 1
    prediction_used: bool = False

    # 原因说明
    reason: str = ""


LEVEL_THRESHOLDS = [(25, 1, "正常", "green"),
                    (45, 2, "轻微关注", "blue"),
                    (62, 3, "需要监控", "yellow"),
                    (78, 4, "高度警惕", "orange"),
                    (101, 5, "紧急预警", "red")]


class OpinionLevelCalculator:
    """
    多维舆情等级评分计算器。

    使用方式：
      calculator = OpinionLevelCalculator(graph)
      score = calculator.calculate_score(snapshot, prev_snapshot, prediction)
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph

    def calculate_score(
        self,
        snapshot: GraphSnapshot,
        prev_snapshot: Optional[GraphSnapshot] = None,
        prediction: Optional[PredictionResult] = None,
    ) -> OpinionScore:
        """
        计算单个快照的综合舆情评分。

        参数：
          snapshot      : 当前快照
          prev_snapshot : 上一快照（用于趋势计算）
          prediction    : KGCN 预测结果（用于前瞻性调整）
        """
        sent_score = self._calc_sentiment_score(snapshot)
        prop_score = self._calc_propagation_score(snapshot)
        inf_score = self._calc_influence_score(snapshot)
        trend_score = self._calc_trend_score(snapshot, prev_snapshot)

        # 加权综合分数
        final = (
            sent_score * 0.40
            + prop_score * 0.25
            + inf_score * 0.20
            + trend_score * 0.15
        )
        final = round(min(100.0, max(0.0, final)), 2)

        level, label, color = self._score_to_level(final)

        # KGCN 预测调整
        adjusted_level = level
        prediction_used = False
        if prediction is not None:
            adjusted_level = max(level, prediction.risk_level)
            if adjusted_level != level:
                prediction_used = True

        reason = self._build_reason(
            snapshot, sent_score, prop_score, inf_score, trend_score,
            final, level, adjusted_level, prediction
        )

        return OpinionScore(
            snapshot_id=snapshot.snapshot_id,
            sentiment_score=round(sent_score, 2),
            propagation_score=round(prop_score, 2),
            influence_score=round(inf_score, 2),
            trend_score=round(trend_score, 2),
            final_score=final,
            level=level,
            level_label=label,
            level_color=color,
            adjusted_level=adjusted_level,
            prediction_used=prediction_used,
            reason=reason,
        )

    def calculate_all_scores(
        self,
        snapshots: List[GraphSnapshot],
        prediction: Optional[PredictionResult] = None,
    ) -> List[OpinionScore]:
        """
        对快照序列中的每个快照计算评分，返回列表。
        """
        scores = []
        for i, snap in enumerate(snapshots):
            prev = snapshots[i - 1] if i > 0 else None
            score = self.calculate_score(snap, prev_snapshot=prev, prediction=prediction)
            scores.append(score)
        return scores

    # ------------------------------------------------------------------
    # 维度计算
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_sentiment_score(snapshot: GraphSnapshot) -> float:
        """
        情感维度得分 [0, 100]。

        公式：
          score = negative_ratio * 60 + extreme_negative_ratio * 40
          直觉：负面比例是主要指标，极端负面是加重因素
        """
        return min(100.0, snapshot.negative_ratio * 60 + snapshot.extreme_negative_ratio * 40)

    @staticmethod
    def _calc_propagation_score(snapshot: GraphSnapshot) -> float:
        """
        传播维度得分 [0, 100]。

        公式：
          participation = num_comments / num_users（人均评论数，衡量热度）
          volume_score  = min(50, num_comments / 1000 * 50)（评论量）
          part_score    = min(50, participation * 10)
          total = volume_score + part_score
        """
        num_comments = snapshot.num_comments
        num_users = max(1, snapshot.num_users)
        participation = num_comments / num_users

        volume_score = min(50.0, num_comments / 1000.0 * 50)
        part_score = min(50.0, participation * 10)
        return volume_score + part_score

    def _calc_influence_score(self, snapshot: GraphSnapshot) -> float:
        """
        影响力维度得分 [0, 100]。

        需要图数据（self.graph），若无则使用快照的 community 信息近似。

        公式：
          kol_score   = certified_users * 5 + high_follower_users * 3
          extra       = 若有 KOL 参与，+20
          virality    = avg_virality * 50
        """
        if self.graph is None:
            return min(100.0, snapshot.avg_virality * 100)

        certified = sum(
            1 for _, d in self.graph.nodes(data=True)
            if d.get("node_type") == NodeType.USER and d.get("is_certified", False)
        )
        high_followers = sum(
            1 for _, d in self.graph.nodes(data=True)
            if d.get("node_type") == NodeType.USER
            and int(d.get("followers_count", 0)) > 100_000
        )
        kol_score = min(50.0, certified * 5 + high_followers * 3)
        virality_score = min(50.0, snapshot.avg_virality * 50)
        return kol_score + virality_score

    @staticmethod
    def _calc_trend_score(
        snapshot: GraphSnapshot, prev_snapshot: Optional[GraphSnapshot]
    ) -> float:
        """
        趋势维度得分 [0, 100]。

        公式：
          若 sentiment_delta < 0（情感在下降），得分升高：
          score = max(0, -sentiment_delta * 200)
          若无前一快照，得分 = 0
        """
        if prev_snapshot is None or snapshot.sentiment_delta == 0.0:
            return 0.0
        return min(100.0, max(0.0, -snapshot.sentiment_delta * 200))

    @staticmethod
    def _score_to_level(score: float):
        for threshold, level, label, color in LEVEL_THRESHOLDS:
            if score < threshold:
                return level, label, color
        return 5, "紧急预警", "red"

    @staticmethod
    def _build_reason(
        snapshot, sent_score, prop_score, inf_score, trend_score,
        final, level, adjusted_level, prediction
    ) -> str:
        lines = [
            f"快照：{snapshot.snapshot_id}",
            f"  情感分：{sent_score:.1f}  传播分：{prop_score:.1f}  "
            f"影响力分：{inf_score:.1f}  趋势分：{trend_score:.1f}",
            f"  综合分：{final:.1f}  等级：L{level}（{OpinionLevelCalculator._score_to_level(final)[1]}）",
        ]
        if snapshot.has_anomaly:
            lines.append(f"  ⚠️ 异常信号：极端负面率={snapshot.extreme_negative_ratio:.1%}")
        if snapshot.sentiment_delta < -0.05:
            lines.append(f"  📉 情感急速下降：Δ={snapshot.sentiment_delta:.3f}")
        if adjusted_level > level and prediction:
            lines.append(
                f"  🔮 KGCN 预测调整等级 L{level} → L{adjusted_level}"
                f"（预测置信度={prediction.confidence:.1%}）"
            )
        return "\n".join(lines)
