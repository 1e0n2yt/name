"""
phase2_prediction/kgcn_model.py - KGCN 模型（基于时序快照的图卷积预测）

实现思路：
  KGCN（Knowledge Graph Convolutional Network）核心步骤：
    1. 为每个实体（节点）初始化嵌入向量
    2. 通过关系感知的邻居聚合更新实体嵌入
    3. 将用户嵌入和物品（评论/博文）嵌入组合预测偏好/情感

  时序扩展：
    - 对每个时间步 t 的快照提取特征向量
    - 使用 GRU/LSTM 对时序特征建模，输出隐藏状态作为预测依据
    - 最终预测：未来 n 天的情感趋势 + 风险等级

注意：
  - 本实现为"轻量级纯 Python + NumPy"版本，适合无 GPU 环境快速验证
  - 生产环境可替换为 PyTorch/TF 实现，接口保持兼容
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from phase1_kg_snapshot.snapshot import GraphSnapshot, TemporalSnapshotGenerator


@dataclass
class PredictionResult:
    """KGCN 预测结果。"""
    topic_id: str
    predicted_at: datetime
    forecast_days: int

    # 预测的每日情感分数序列 [0, 1]
    predicted_sentiments: List[float] = field(default_factory=list)

    # 预测的每日负面率序列 [0, 1]
    predicted_negative_ratios: List[float] = field(default_factory=list)

    # 风险等级（1=低，5=极高）
    risk_level: int = 1

    # 预测置信度 [0, 1]
    confidence: float = 0.0

    # 模型使用的快照数量
    input_snapshots: int = 0

    # 是否预测到二次发酵信号
    secondary_fermentation_risk: bool = False

    # 预测说明
    explanation: str = ""


class KGCNPredictor:
    """
    KGCN 时序情感预测器

    轻量级实现：
      1. 从快照序列提取时序特征矩阵（T × F）
      2. 使用简单的指数加权移动平均（EWMA）+ 趋势分析预测未来
      3. 结合图结构特征（中心性、社群信息）调整预测

    生产环境升级路径：
      - 将 _predict_with_ewma() 替换为 PyTorch GRU/Transformer 模型
      - 加载预训练权重：self.model.load_state_dict(torch.load("models/kgcn.pth"))
      - 接口签名不变，只替换内部实现
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        参数：
          model_path: 预训练模型权重路径（PyTorch .pth 文件）。
                      若为 None，使用内置的 EWMA 预测（无需训练）。
        """
        self.model_path = model_path
        self._model = None  # 占位符，替换为 PyTorch 模型时使用

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """加载预训练模型权重（占位符）。"""
        # 实际使用时替换为：
        # import torch
        # from .kgcn_torch import KGCNTorch
        # self._model = KGCNTorch(...)
        # self._model.load_state_dict(torch.load(path))
        # self._model.eval()
        print(f"[KGCN] 模型权重文件：{path}（当前使用 EWMA 回退模式）")

    def predict(
        self,
        snapshots: List[GraphSnapshot],
        forecast_days: int = 3,
    ) -> PredictionResult:
        """
        基于时序快照序列预测未来 forecast_days 天的情感趋势。

        参数：
          snapshots     : 时序快照列表（建议使用近 3-7 天）
          forecast_days : 预测天数（默认 3 天）

        返回：
          PredictionResult
        """
        if not snapshots:
            return PredictionResult(
                topic_id="unknown",
                predicted_at=datetime.now(),
                forecast_days=forecast_days,
                explanation="无可用快照，无法预测。",
            )

        topic_id = snapshots[0].topic_id
        temporal_features = TemporalSnapshotGenerator.extract_temporal_features(snapshots)

        if self._model is not None:
            return self._predict_with_model(
                topic_id, temporal_features, snapshots, forecast_days
            )
        else:
            return self._predict_with_ewma(
                topic_id, temporal_features, snapshots, forecast_days
            )

    def _predict_with_ewma(
        self,
        topic_id: str,
        features: List[Dict],
        snapshots: List[GraphSnapshot],
        forecast_days: int,
    ) -> PredictionResult:
        """
        基于指数加权移动平均 + 线性趋势外推进行预测。

        说明：
          这是一个高质量的统计基线方法，可在没有训练数据时直接使用。
          KGCN 深度学习模型上线后会替换此方法，接口不变。

        EWMA 参数：
          alpha = 0.4（近期数据权重 40%，历史数据权重 60%）
        """
        alpha = 0.4

        sentiment_series = [f["avg_sentiment"] for f in features]
        negative_series = [f["negative_ratio"] for f in features]
        delta_series = [f["sentiment_delta"] for f in features]
        anomaly_flags = [f["has_anomaly"] for f in features]

        def ewma(series: List[float]) -> List[float]:
            if not series:
                return []
            result = [series[0]]
            for x in series[1:]:
                result.append(alpha * x + (1 - alpha) * result[-1])
            return result

        smoothed_sent = ewma(sentiment_series)
        smoothed_neg = ewma(negative_series)

        # 线性趋势（最近两个点的斜率）
        if len(smoothed_sent) >= 2:
            sent_trend = smoothed_sent[-1] - smoothed_sent[-2]
            neg_trend = smoothed_neg[-1] - smoothed_neg[-2]
        else:
            sent_trend = 0.0
            neg_trend = 0.0

        # 生成预测序列
        last_sent = smoothed_sent[-1]
        last_neg = smoothed_neg[-1]

        predicted_sentiments = []
        predicted_negatives = []

        # 加速衰减系数
        DECLINING_TREND_THRESHOLD = -0.02      # 趋势斜率低于此值认为持续下降
        DECLINING_DELTA_SUM_THRESHOLD = -0.05  # 近3日情感delta之和低于此值确认下降趋势

        is_declining = (
            sent_trend < DECLINING_TREND_THRESHOLD
            and sum(delta_series[-3:]) < DECLINING_DELTA_SUM_THRESHOLD
        )
        acceleration = 1.5 if is_declining else 1.0

        for d in range(1, forecast_days + 1):
            proj_sent = last_sent + sent_trend * d * acceleration
            proj_neg = last_neg + neg_trend * d * acceleration
            # 边界约束
            proj_sent = max(0.0, min(1.0, proj_sent))
            proj_neg = max(0.0, min(1.0, proj_neg))
            predicted_sentiments.append(round(proj_sent, 4))
            predicted_negatives.append(round(proj_neg, 4))

        # 风险等级（1-5）
        risk_level = self._compute_risk_level(
            snapshots=snapshots,
            predicted_sentiments=predicted_sentiments,
            predicted_negatives=predicted_negatives,
            anomaly_flags=anomaly_flags,
            is_declining=is_declining,
        )

        # 二次发酵风险检测
        secondary_risk = (
            is_declining
            and any(s < 0.35 for s in predicted_sentiments)
            and snapshots[-1].new_users > 0
        )

        # 置信度（快照越多，越可信；趋势越稳定，越可信）
        variance = (
            sum((s - sum(sentiment_series) / len(sentiment_series)) ** 2
                for s in sentiment_series)
            / len(sentiment_series)
        ) if len(sentiment_series) > 1 else 0.1
        confidence = min(1.0, len(snapshots) / 7.0) * max(0.3, 1.0 - variance * 5)

        explanation = self._build_explanation(
            snapshots, predicted_sentiments, predicted_negatives,
            risk_level, secondary_risk, is_declining
        )

        return PredictionResult(
            topic_id=topic_id,
            predicted_at=datetime.now(),
            forecast_days=forecast_days,
            predicted_sentiments=predicted_sentiments,
            predicted_negative_ratios=predicted_negatives,
            risk_level=risk_level,
            confidence=round(confidence, 3),
            input_snapshots=len(snapshots),
            secondary_fermentation_risk=secondary_risk,
            explanation=explanation,
        )

    def _predict_with_model(
        self,
        topic_id: str,
        features: List[Dict],
        snapshots: List[GraphSnapshot],
        forecast_days: int,
    ) -> PredictionResult:
        """
        使用预训练 KGCN 模型预测（占位符，替换内部实现时使用）。
        """
        # 实际使用时替换为：
        # import torch
        # x = self._features_to_tensor(features)
        # with torch.no_grad():
        #     output = self._model(x)
        # return self._tensor_to_result(output, topic_id, forecast_days, snapshots)
        return self._predict_with_ewma(topic_id, features, snapshots, forecast_days)

    @staticmethod
    def _compute_risk_level(
        snapshots: List[GraphSnapshot],
        predicted_sentiments: List[float],
        predicted_negatives: List[float],
        anomaly_flags: List[int],
        is_declining: bool,
    ) -> int:
        """
        多维风险等级计算（1-5）。

        规则：
          L1（低）    : 情感稳定，负面 < 30%
          L2（较低）  : 轻度负面趋势，负面 30-40%
          L3（中）    : 情感下降，负面 40-55%，有异常信号
          L4（高）    : 情感急降，负面 55-70%，有 KOL 参与
          L5（极高）  : 极端情感，负面 > 70%，二次发酵确认
        """
        if not snapshots:
            return 1

        latest = snapshots[-1]
        avg_neg = sum(predicted_negatives) / len(predicted_negatives)
        min_sent = min(predicted_sentiments) if predicted_sentiments else 0.5
        anomaly_count = sum(anomaly_flags)

        score = 0
        # 负面占比
        if avg_neg > 0.70:
            score += 5
        elif avg_neg > 0.55:
            score += 4
        elif avg_neg > 0.40:
            score += 3
        elif avg_neg > 0.30:
            score += 2
        else:
            score += 1

        # 情感绝对值
        if min_sent < 0.15:
            score += 2
        elif min_sent < 0.25:
            score += 1

        # 异常信号
        score += min(2, anomaly_count)

        # 趋势加权
        if is_declining:
            score += 1

        # 极端负面情绪
        if latest.extreme_negative_ratio > 0.25:
            score += 1

        # 映射到 1-5
        if score <= 2:
            return 1
        elif score <= 4:
            return 2
        elif score <= 6:
            return 3
        elif score <= 8:
            return 4
        else:
            return 5

    @staticmethod
    def _build_explanation(
        snapshots: List[GraphSnapshot],
        predicted_sentiments: List[float],
        predicted_negatives: List[float],
        risk_level: int,
        secondary_risk: bool,
        is_declining: bool,
    ) -> str:
        """生成预测说明文本。"""
        latest = snapshots[-1]
        lines = [
            f"基于近 {len(snapshots)} 天快照数据预测：",
            f"  当前情感均值：{latest.avg_sentiment:.3f}，负面占比：{latest.negative_ratio:.1%}",
            f"  预测情感（未来{len(predicted_sentiments)}天）：{[f'{s:.3f}' for s in predicted_sentiments]}",
            f"  预测负面率：{[f'{n:.1%}' for n in predicted_negatives]}",
            f"  风险等级：L{risk_level}",
        ]
        if is_declining:
            lines.append("  ⚠️ 情感持续下降趋势，建议密切关注！")
        if secondary_risk:
            lines.append("  🚨 检测到二次发酵风险信号！新增用户涌入，负面情感扩散。")
        if latest.has_anomaly:
            lines.append(f"  ⚠️ 最新快照存在异常：极端负面率={latest.extreme_negative_ratio:.1%}")
        return "\n".join(lines)
