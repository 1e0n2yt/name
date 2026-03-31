"""
phase2_prediction/alert_system.py - 舆情预警生成系统

预警级别：
  L1 绿色 - 正常，无预警
  L2 蓝色 - 轻微，建议定时关注
  L3 黄色 - 一般预警，通知相关负责人
  L4 橙色 - 重要预警，启动应急预案
  L5 红色 - 紧急预警，立即响应

预警触发条件：
  1. 综合评分超过阈值
  2. 情感急速下降（Δ < -0.05）
  3. 极端负面率超标（> 30%）
  4. 新用户涌入异常（new_users / active_users > 40%）
  5. KGCN 预测二次发酵风险
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from phase1_kg_snapshot.snapshot import GraphSnapshot
from phase2_prediction.kgcn_model import PredictionResult
from phase2_prediction.opinion_level import OpinionScore


@dataclass
class Alert:
    """单条预警记录。"""
    alert_id: str
    topic_id: str
    level: int              # 1-5
    level_label: str        # 对应颜色/标签
    trigger_reason: str     # 触发原因
    recommended_action: str # 建议行动
    snapshot_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # 附加数据
    current_sentiment: float = 0.5
    predicted_sentiment_3d: Optional[float] = None  # 3天后预测情感
    secondary_risk: bool = False


ALERT_LABELS = {
    1: ("绿色", "正常"),
    2: ("蓝色", "轻微关注"),
    3: ("黄色", "一般预警"),
    4: ("橙色", "重要预警"),
    5: ("红色", "紧急预警"),
}

ACTIONS = {
    1: "继续常规监控，每日检查一次。",
    2: "适当增加监控频率，每6小时检查一次。",
    3: "通知相关负责人，密切跟踪情感走势，准备应对素材。",
    4: "启动应急预案，安排公关团队响应，发布官方声明，监控全网扩散。",
    5: "立即响应！召开紧急会议，全面启动危机管理流程，协调法务/公关/运营联合处置。",
}


class AlertSystem:
    """
    舆情预警生成系统。

    使用方式：
      alert_system = AlertSystem()
      alerts = alert_system.generate_alerts(snapshots, scores, prediction)
    """

    def generate_alerts(
        self,
        snapshots: List[GraphSnapshot],
        scores: List[OpinionScore],
        prediction: Optional[PredictionResult] = None,
    ) -> List[Alert]:
        """
        综合快照统计、评分结果和 KGCN 预测生成预警列表。

        参数：
          snapshots  : 时序快照列表
          scores     : 对应评分列表
          prediction : KGCN 预测结果（可选）

        返回：
          Alert 列表（L2 及以上才生成预警）
        """
        alerts: List[Alert] = []

        # 对最近快照生成预警
        if snapshots and scores:
            latest_snap = snapshots[-1]
            latest_score = scores[-1]
            alert = self._check_snapshot(latest_snap, latest_score, prediction)
            if alert:
                alerts.append(alert)

        # 二次发酵专项预警
        if prediction and prediction.secondary_fermentation_risk:
            alerts.append(
                self._build_secondary_fermentation_alert(
                    topic_id=snapshots[-1].topic_id if snapshots else "unknown",
                    snapshot=snapshots[-1] if snapshots else None,
                    prediction=prediction,
                )
            )

        return alerts

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _check_snapshot(
        self,
        snapshot: GraphSnapshot,
        score: OpinionScore,
        prediction: Optional[PredictionResult],
    ) -> Optional[Alert]:
        """
        根据快照和评分检查是否需要生成预警。
        只在 adjusted_level >= 2 时生成预警。
        """
        level = score.adjusted_level
        if level < 2:
            return None

        color, label = ALERT_LABELS[level]
        reasons = []

        if snapshot.extreme_negative_ratio > 0.30:
            reasons.append(f"极端负面率 {snapshot.extreme_negative_ratio:.1%} 超过警戒线（30%）")
        if snapshot.sentiment_delta < -0.05:
            reasons.append(f"情感急速下降 Δ={snapshot.sentiment_delta:.3f}")
        if snapshot.has_anomaly:
            reasons.append("检测到情感异常信号")
        if snapshot.new_users > snapshot.active_users * 0.4 and snapshot.active_users > 0:
            reasons.append(
                f"新增用户涌入比例 {snapshot.new_users/snapshot.active_users:.1%} 异常"
            )
        if score.prediction_used:
            reasons.append(f"KGCN预测评级上调为 L{level}")

        if not reasons:
            reasons.append(f"综合评分达到 {score.final_score:.1f}")

        pred_3d = None
        if prediction and prediction.predicted_sentiments:
            pred_3d = prediction.predicted_sentiments[-1]

        return Alert(
            alert_id=f"alert_{snapshot.snapshot_id}_{level}",
            topic_id=snapshot.topic_id,
            level=level,
            level_label=f"{color}-{label}",
            trigger_reason=" | ".join(reasons),
            recommended_action=ACTIONS[level],
            snapshot_id=snapshot.snapshot_id,
            current_sentiment=snapshot.avg_sentiment,
            predicted_sentiment_3d=pred_3d,
            secondary_risk=prediction.secondary_fermentation_risk
            if prediction
            else False,
        )

    @staticmethod
    def _build_secondary_fermentation_alert(
        topic_id: str,
        snapshot: Optional[GraphSnapshot],
        prediction: PredictionResult,
    ) -> Alert:
        """二次发酵专项预警（L4 起步）。"""
        level = max(4, prediction.risk_level)
        color, label = ALERT_LABELS[level]

        reason = (
            "KGCN 预测到二次发酵风险信号：情感持续下降 + 新用户涌入 + 负面情感扩散"
        )
        snap_id = snapshot.snapshot_id if snapshot else "N/A"

        return Alert(
            alert_id=f"alert_secondary_{topic_id}_{level}",
            topic_id=topic_id,
            level=level,
            level_label=f"{color}-{label}",
            trigger_reason=reason,
            recommended_action=ACTIONS[level],
            snapshot_id=snap_id,
            current_sentiment=snapshot.avg_sentiment if snapshot else 0.5,
            predicted_sentiment_3d=prediction.predicted_sentiments[-1]
            if prediction.predicted_sentiments
            else None,
            secondary_risk=True,
        )

    @staticmethod
    def format_alerts(alerts: List[Alert]) -> str:
        """格式化输出预警列表。"""
        if not alerts:
            return "✅ 无预警，话题情感正常。"
        lines = [f"\n{'='*60}", "📢 舆情预警报告", f"{'='*60}"]
        for a in alerts:
            lines += [
                f"\n🔔 [{a.level_label}] 话题：{a.topic_id}",
                f"   时间：{a.created_at.strftime('%Y-%m-%d %H:%M')}",
                f"   触发原因：{a.trigger_reason}",
                f"   当前情感均值：{a.current_sentiment:.3f}",
            ]
            if a.predicted_sentiment_3d is not None:
                lines.append(f"   3天后预测情感：{a.predicted_sentiment_3d:.3f}")
            if a.secondary_risk:
                lines.append("   🚨 二次发酵风险！")
            lines.append(f"   建议行动：{a.recommended_action}")
        return "\n".join(lines)
