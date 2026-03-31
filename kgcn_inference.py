"""
kgcn_inference.py
KGCN 模型推理与异常检测模块（代码3）

功能：
    - 加载本地保存的 KGCN 模型权重
    - 通过话题名称访问保存在本地的知识图谱时序快照
    - 基于 KGCN 模型预测未来情感趋势
    - 输出预测结果：未来 n 天的情感分布
    - 输出风险等级（1-5 级）
    - 检测异常信号：
        * 情感激增（负面情感占比上升 > 15%）
        * 高风险用户激增（一天内新增高风险用户数 >= 2）
        * 用户情感快速恶化（同一用户情感分数快速下降）
        * 评论数激增（一天内评论数增长 > 200%）

输入数据格式：
    - topic_name: 话题名称
    - model_path: 模型权重路径（.pth）
    - config_path: 模型配置路径（.json）
    - forecast_days: 预测天数（默认 3）

输出数据格式：
    预测结果字典，包含：
    - predictions: 未来 n 天的情感分布列表
    - risk_levels: 对应的舆情等级列表
    - anomalies: 检测到的异常信号列表
    - confidence: 预测的置信度

遵循 PEP8 规范，random_state=42 保证可复现性。
"""

import json
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from kg_snapshot_builder import KGSnapshot, NodeType, TemporalSnapshotManager
from kgcn_trainer import SentimentKGCN, compute_risk_levels, predict_future_sentiment
from opinion_level_calc import (
    OpinionLevelResult,
    calculate_opinion_level_from_distribution,
)

# 全局随机种子
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 异常信号检测
# ──────────────────────────────────────────────────────────────────────────────

class AnomalyType:
    """异常类型常量"""
    SENTIMENT_SPIKE = "sentiment_spike"          # 情感激增
    HIGH_RISK_USER_SURGE = "high_risk_user_surge"  # 高风险用户激增
    USER_RAPID_DETERIORATION = "user_rapid_deterioration"  # 用户情感快速恶化
    COMMENT_COUNT_SURGE = "comment_count_surge"  # 评论数激增


def detect_sentiment_spike(
    snapshots: List[KGSnapshot],
    threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    检测情感激增异常（负面情感占比上升 > threshold）

    对每相邻两天的快照进行对比，若负面情感比例变化超过阈值则触发异常。

    Args:
        snapshots (List[KGSnapshot]): 快照列表（按时间升序）
        threshold (float): 负面情感上升阈值，默认 0.15（15%）

    Returns:
        List[Dict]: 检测到的情感激增异常列表，每项包含：
            - type: 异常类型
            - date: 发生日期
            - severity: 严重程度 [0, 1]
            - description: 异常描述
            - neg_ratio_prev: 前一天负面比例
            - neg_ratio_curr: 当天负面比例
            - delta: 负面比例变化量
    """
    anomalies = []
    for i in range(1, len(snapshots)):
        prev_snap = snapshots[i - 1]
        curr_snap = snapshots[i]
        neg_prev = prev_snap.sentiment_distribution["negative"]
        neg_curr = curr_snap.sentiment_distribution["negative"]
        delta = neg_curr - neg_prev

        if delta > threshold:
            # 严重程度：超过阈值越多越严重，最高 1.0
            severity = min(1.0, delta / 0.5)
            anomalies.append({
                "type": AnomalyType.SENTIMENT_SPIKE,
                "date": curr_snap.date.strftime("%Y-%m-%d"),
                "severity": round(severity, 3),
                "description": (
                    f"负面情感快速上升 {delta:.1%}：{neg_prev:.1%} → {neg_curr:.1%}"
                ),
                "neg_ratio_prev": round(neg_prev, 4),
                "neg_ratio_curr": round(neg_curr, 4),
                "delta": round(delta, 4),
            })
            logger.info(
                "检测到情感激增：%s 负面+%.1f%%",
                curr_snap.date.strftime("%Y-%m-%d"), delta * 100,
            )

    return anomalies


def detect_high_risk_user_surge(
    snapshots: List[KGSnapshot],
    min_new_users: int = 2,
) -> List[Dict[str, Any]]:
    """
    检测高风险用户激增异常（一天内新增高风险用户数 >= min_new_users）

    Args:
        snapshots (List[KGSnapshot]): 快照列表（按时间升序）
        min_new_users (int): 新增高风险用户数阈值，默认 2

    Returns:
        List[Dict]: 检测到的高风险用户激增异常列表，每项包含：
            - type: 异常类型
            - date: 发生日期
            - severity: 严重程度 [0, 1]
            - description: 异常描述
            - prev_high_risk_count: 前一天高风险用户数
            - curr_high_risk_count: 当天高风险用户数
            - new_high_risk_users: 新增的高风险用户 ID 列表
    """
    anomalies = []
    for i in range(1, len(snapshots)):
        prev_snap = snapshots[i - 1]
        curr_snap = snapshots[i]

        prev_set = set(prev_snap.high_risk_user_ids)
        curr_set = set(curr_snap.high_risk_user_ids)
        new_high_risk = curr_set - prev_set

        if len(new_high_risk) >= min_new_users:
            severity = min(1.0, len(new_high_risk) / 10.0)
            anomalies.append({
                "type": AnomalyType.HIGH_RISK_USER_SURGE,
                "date": curr_snap.date.strftime("%Y-%m-%d"),
                "severity": round(severity, 3),
                "description": (
                    f"新增 {len(new_high_risk)} 个高风险用户"
                    f"（共 {len(curr_set)} 个）"
                ),
                "prev_high_risk_count": len(prev_set),
                "curr_high_risk_count": len(curr_set),
                "new_high_risk_users": list(new_high_risk),
            })
            logger.info(
                "检测到高风险用户激增：%s 新增 %d 人",
                curr_snap.date.strftime("%Y-%m-%d"), len(new_high_risk),
            )

    return anomalies


def detect_user_rapid_deterioration(
    snapshots: List[KGSnapshot],
    min_deterioration: float = 0.3,
    min_occurrences: int = 2,
) -> List[Dict[str, Any]]:
    """
    检测用户情感快速恶化异常（同一用户情感分数快速下降）

    从快照图中提取用户的评论情感分数，统计在近期快照中情感快速下降的用户。

    Args:
        snapshots (List[KGSnapshot]): 快照列表（按时间升序）
        min_deterioration (float): 单次情感下降幅度阈值，默认 0.3
        min_occurrences (int): 需要连续恶化的次数，默认 2

    Returns:
        List[Dict]: 检测到的用户情感恶化异常列表，每项包含：
            - type: 异常类型
            - date: 最近检测日期
            - severity: 严重程度
            - description: 异常描述
            - user_id: 恶化的用户 ID
            - sentiment_trajectory: 情感轨迹（从早到晚）
    """
    # 收集每个用户在各快照中的平均情感分数
    user_sentiments: Dict[str, List[float]] = {}
    user_last_date: Dict[str, str] = {}

    for snap in snapshots:
        date_str = snap.date.strftime("%Y-%m-%d")
        user_scores: Dict[str, List[float]] = {}

        for _, data in snap.graph.nodes(data=True):
            if data.get("node_type") == NodeType.COMMENT:
                uid = data.get("user_id", "")
                score = data.get("sentiment_score", 0.0)
                if uid:
                    user_scores.setdefault(uid, []).append(score)

        for uid, scores in user_scores.items():
            avg_score = sum(scores) / len(scores)
            user_sentiments.setdefault(uid, []).append(avg_score)
            user_last_date[uid] = date_str

    # 检测连续恶化的用户
    anomalies = []
    for uid, scores in user_sentiments.items():
        if len(scores) < min_occurrences + 1:
            continue

        consecutive_drops = 0
        for i in range(1, len(scores)):
            drop = scores[i - 1] - scores[i]  # 正值表示下降
            if drop >= min_deterioration:
                consecutive_drops += 1
            else:
                consecutive_drops = 0

            if consecutive_drops >= min_occurrences:
                # 严重程度：总下降幅度
                total_drop = scores[0] - scores[-1]
                severity = min(1.0, total_drop / 1.5)
                anomalies.append({
                    "type": AnomalyType.USER_RAPID_DETERIORATION,
                    "date": user_last_date.get(uid, "unknown"),
                    "severity": round(max(0.0, severity), 3),
                    "description": (
                        f"用户 {uid} 情感快速恶化：{scores[0]:.2f} → {scores[-1]:.2f}"
                        f"（下降 {total_drop:.2f}）"
                    ),
                    "user_id": uid,
                    "sentiment_trajectory": [round(s, 3) for s in scores],
                })
                logger.info(
                    "检测到用户情感恶化：用户 %s 情感从 %.2f → %.2f",
                    uid, scores[0], scores[-1],
                )
                break  # 每个用户只报告一次

    return anomalies


def detect_comment_count_surge(
    snapshots: List[KGSnapshot],
    growth_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    检测评论数激增异常（一天内评论数增长 > growth_threshold * 100%）

    Args:
        snapshots (List[KGSnapshot]): 快照列表（按时间升序）
        growth_threshold (float): 增长倍数阈值（2.0 = 增长 200%），默认 2.0

    Returns:
        List[Dict]: 检测到的评论激增异常列表，每项包含：
            - type: 异常类型
            - date: 发生日期
            - severity: 严重程度
            - description: 异常描述
            - prev_count: 前一天评论数
            - curr_count: 当天评论数
            - growth_rate: 增长率
    """
    anomalies = []
    for i in range(1, len(snapshots)):
        prev_snap = snapshots[i - 1]
        curr_snap = snapshots[i]
        prev_count = prev_snap.comment_count
        curr_count = curr_snap.comment_count

        if prev_count == 0:
            continue

        growth_rate = (curr_count - prev_count) / prev_count
        if growth_rate > growth_threshold:
            severity = min(1.0, growth_rate / 5.0)
            anomalies.append({
                "type": AnomalyType.COMMENT_COUNT_SURGE,
                "date": curr_snap.date.strftime("%Y-%m-%d"),
                "severity": round(severity, 3),
                "description": (
                    f"评论数激增 {growth_rate:.1%}：{prev_count} → {curr_count} 条"
                ),
                "prev_count": prev_count,
                "curr_count": curr_count,
                "growth_rate": round(growth_rate, 4),
            })
            logger.info(
                "检测到评论激增：%s 增长 %.1f%%（%d → %d）",
                curr_snap.date.strftime("%Y-%m-%d"),
                growth_rate * 100, prev_count, curr_count,
            )

    return anomalies


def detect_all_anomalies(
    snapshots: List[KGSnapshot],
    sentiment_spike_threshold: float = 0.15,
    high_risk_user_min_new: int = 2,
    user_deterioration_drop: float = 0.3,
    comment_surge_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    综合检测所有类型的舆情异常信号

    Args:
        snapshots (List[KGSnapshot]): 快照列表（按时间升序）
        sentiment_spike_threshold (float): 情感激增阈值，默认 0.15
        high_risk_user_min_new (int): 高风险用户激增最小新增数，默认 2
        user_deterioration_drop (float): 用户情感恶化单次下降幅度阈值，默认 0.3
        comment_surge_threshold (float): 评论激增倍数阈值，默认 2.0

    Returns:
        List[Dict]: 所有异常信号列表（按严重程度降序排列）
    """
    all_anomalies = []

    # 检测各类异常
    all_anomalies.extend(detect_sentiment_spike(snapshots, threshold=sentiment_spike_threshold))
    all_anomalies.extend(detect_high_risk_user_surge(snapshots, min_new_users=high_risk_user_min_new))
    all_anomalies.extend(detect_user_rapid_deterioration(snapshots, min_deterioration=user_deterioration_drop))
    all_anomalies.extend(detect_comment_count_surge(snapshots, growth_threshold=comment_surge_threshold))

    # 按严重程度降序排列
    all_anomalies.sort(key=lambda x: x["severity"], reverse=True)
    logger.info("共检测到 %d 个异常信号", len(all_anomalies))
    return all_anomalies


# ──────────────────────────────────────────────────────────────────────────────
# 置信度计算
# ──────────────────────────────────────────────────────────────────────────────

def _compute_prediction_confidence(
    snapshots: List[KGSnapshot],
    predictions: List[Dict[str, float]],
) -> float:
    """
    计算预测置信度

    置信度基于以下因素：
    1. 历史数据量（快照越多置信度越高）
    2. 历史情感趋势的稳定性（趋势越稳定置信度越高）
    3. 预测值与历史趋势的一致性

    Args:
        snapshots (List[KGSnapshot]): 历史快照列表
        predictions (List[Dict[str, float]]): 预测的情感分布列表

    Returns:
        float: 置信度，范围 [0, 1]
    """
    if not snapshots or not predictions:
        return 0.0

    # 因素1：数据量置信度（7 天为满分）
    data_confidence = min(1.0, len(snapshots) / 7.0)

    # 因素2：历史趋势稳定性（负面情感方差越小越稳定）
    neg_ratios = [s.sentiment_distribution["negative"] for s in snapshots]
    if len(neg_ratios) > 1:
        variance = float(np.var(neg_ratios))
        stability_confidence = max(0.0, 1.0 - variance * 4)
    else:
        stability_confidence = 0.5

    # 因素3：预测一致性（预测负面趋势是否符合历史趋势方向）
    if len(neg_ratios) >= 2:
        hist_trend = neg_ratios[-1] - neg_ratios[-2]  # 历史最后两天的变化
        pred_trend = predictions[0]["negative"] - neg_ratios[-1]
        trend_consistency = 1.0 - min(1.0, abs(hist_trend - pred_trend) * 5)
    else:
        trend_consistency = 0.5

    # 加权综合
    confidence = (
        0.4 * data_confidence
        + 0.3 * stability_confidence
        + 0.3 * trend_consistency
    )
    return float(max(0.0, min(1.0, confidence)))


# ──────────────────────────────────────────────────────────────────────────────
# 主推理类
# ──────────────────────────────────────────────────────────────────────────────

class KGCNPredictor:
    """
    KGCN 舆情预测推理器

    加载训练好的 KGCN 模型，对指定话题执行推理：
    - 预测未来 n 天情感分布
    - 计算舆情风险等级
    - 检测异常信号

    Usage:
        predictor = KGCNPredictor(
            model_path="models/topic/kgcn_xxx.pth",
            config_path="models/topic/kgcn_xxx.json",
            snapshot_dir="./snapshots",
        )
        result = predictor.predict("某热点话题", forecast_days=3)
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        snapshot_dir: str = "./snapshots",
    ):
        """
        初始化推理器，加载模型和配置

        Args:
            model_path (str): 模型权重文件路径（.pth）
            config_path (str): 模型配置文件路径（.json）
            snapshot_dir (str): 快照根目录

        Raises:
            FileNotFoundError: 模型文件或配置文件不存在
            ValueError: 配置文件格式错误
        """
        # 验证文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件不存在：{model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置文件不存在：{config_path}")

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        required_keys = [
            "node_feature_dim", "hidden_dim", "lstm_hidden_dim",
            "num_kgcn_layers", "lstm_layers", "forecast_days", "seq_len",
        ]
        missing = [k for k in required_keys if k not in self.config]
        if missing:
            raise ValueError(f"配置文件缺少必要键：{missing}")

        # 实例化模型
        self.model = SentimentKGCN(
            node_feature_dim=self.config["node_feature_dim"],
            hidden_dim=self.config["hidden_dim"],
            lstm_hidden_dim=self.config["lstm_hidden_dim"],
            num_kgcn_layers=self.config["num_kgcn_layers"],
            lstm_layers=self.config["lstm_layers"],
            forecast_days=self.config["forecast_days"],
        ).to(DEVICE)

        # 加载权重
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.seq_len = self.config["seq_len"]
        self.snapshot_manager = TemporalSnapshotManager(snapshot_dir=snapshot_dir)

        logger.info(
            "模型加载完成：%s（seq_len=%d, forecast_days=%d）",
            Path(model_path).name,
            self.seq_len,
            self.config["forecast_days"],
        )

    def predict(
        self,
        topic_name: str,
        forecast_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        对指定话题执行舆情预测与异常检测

        Args:
            topic_name (str): 话题名称（用于加载对应的时序快照）
            forecast_days (Optional[int]): 预测天数，若为 None 则使用模型配置中的值

        Returns:
            Dict[str, Any]: 预测结果字典，包含：
                - topic_name (str): 话题名称
                - snapshot_count (int): 使用的快照数量
                - forecast_days (int): 预测天数
                - predictions (List[Dict]): 未来 n 天的情感分布列表，
                    每项格式：
                    {
                        'date': 'YYYY-MM-DD',
                        'positive': float,
                        'neutral': float,
                        'negative': float,
                    }
                - risk_levels (List[Dict]): 对应的舆情等级列表，
                    每项格式：
                    {
                        'date': 'YYYY-MM-DD',
                        'level': int (1-5),
                        'color': str,
                        'composite_score': float,
                        'description': str,
                    }
                - anomalies (List[Dict]): 检测到的异常信号列表
                - confidence (float): 预测置信度，范围 [0, 1]
                - predicted_at (str): 预测执行时间（ISO 格式）
                - history (List[Dict]): 历史快照情感分布（供参考）
        """
        # 1. 加载时序快照
        snapshots = self.snapshot_manager.load_snapshots(topic_name)
        if not snapshots:
            raise ValueError(f"话题「{topic_name}」没有可用的快照，请先执行 build_snapshots()")

        logger.info("话题「%s」：加载了 %d 个快照", topic_name, len(snapshots))

        # 2. 确定预测天数
        n_forecast = forecast_days if forecast_days is not None else self.config["forecast_days"]

        # 若请求的预测天数与模型不同，记录警告
        if n_forecast != self.config["forecast_days"]:
            logger.warning(
                "请求预测 %d 天，但模型训练时配置为 %d 天，将使用模型配置",
                n_forecast, self.config["forecast_days"],
            )
            n_forecast = self.config["forecast_days"]

        # 3. KGCN 预测情感分布
        future_distributions = predict_future_sentiment(
            model=self.model,
            snapshots=snapshots,
            seq_len=self.seq_len,
        )

        # 4. 计算各天风险等级
        risk_results = compute_risk_levels(
            predictions=future_distributions,
            prev_snapshot=snapshots[-1],
        )

        # 5. 检测历史快照中的异常信号
        anomalies = detect_all_anomalies(snapshots)

        # 6. 计算置信度
        confidence = _compute_prediction_confidence(snapshots, future_distributions)

        # 7. 生成预测日期序列（从明天开始）
        last_date = snapshots[-1].date
        pred_dates = [
            (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(n_forecast)
        ]

        # 8. 组装返回结果
        predictions_output = []
        for date_str, dist in zip(pred_dates, future_distributions):
            predictions_output.append({
                "date": date_str,
                "positive": round(dist["positive"], 4),
                "neutral": round(dist["neutral"], 4),
                "negative": round(dist["negative"], 4),
            })

        risk_levels_output = []
        for date_str, risk in zip(pred_dates, risk_results):
            risk_levels_output.append({
                "date": date_str,
                "level": risk.level,
                "color": risk.color,
                "composite_score": round(risk.composite_score, 4),
                "description": risk.description,
            })

        history_output = []
        for snap in snapshots:
            dist = snap.sentiment_distribution
            history_output.append({
                "date": snap.date.strftime("%Y-%m-%d"),
                "positive": round(dist["positive"], 4),
                "neutral": round(dist["neutral"], 4),
                "negative": round(dist["negative"], 4),
                "comment_count": snap.comment_count,
                "high_risk_user_count": len(snap.high_risk_user_ids),
            })

        result = {
            "topic_name": topic_name,
            "snapshot_count": len(snapshots),
            "forecast_days": n_forecast,
            "predictions": predictions_output,
            "risk_levels": risk_levels_output,
            "anomalies": anomalies,
            "confidence": round(confidence, 4),
            "predicted_at": datetime.now().isoformat(),
            "history": history_output,
        }

        logger.info(
            "预测完成：未来 %d 天，置信度=%.2f，检测到 %d 个异常",
            n_forecast, confidence, len(anomalies),
        )
        return result

    @classmethod
    def load(
        cls,
        topic_name: str,
        model_dir: str = "./models",
        snapshot_dir: str = "./snapshots",
    ) -> "KGCNPredictor":
        """
        便捷类方法：从话题模型目录加载最新的模型

        自动查找 {model_dir}/{topic_name}/ 下最新的 .pth 和 .json 文件。

        Args:
            topic_name (str): 话题名称
            model_dir (str): 模型根目录
            snapshot_dir (str): 快照根目录

        Returns:
            KGCNPredictor: 加载好模型的推理器实例

        Raises:
            FileNotFoundError: 模型目录不存在或无可用模型文件
        """
        topic_model_dir = Path(model_dir) / topic_name
        if not topic_model_dir.exists():
            raise FileNotFoundError(
                f"话题模型目录不存在：{topic_model_dir}\n"
                f"请先运行 kgcn_trainer.py 训练模型。"
            )

        pth_files = sorted(topic_model_dir.glob("*.pth"), reverse=True)
        json_files = sorted(topic_model_dir.glob("*.json"), reverse=True)

        if not pth_files:
            raise FileNotFoundError(f"在 {topic_model_dir} 中未找到 .pth 模型文件")
        if not json_files:
            raise FileNotFoundError(f"在 {topic_model_dir} 中未找到 .json 配置文件")

        # 取最新的文件（按文件名排序，文件名包含时间戳）
        model_path = str(pth_files[0])
        config_path = str(json_files[0])

        logger.info("加载最新模型：%s", pth_files[0].name)
        return cls(
            model_path=model_path,
            config_path=config_path,
            snapshot_dir=snapshot_dir,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 便捷函数
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    topic_name: str,
    model_path: str,
    config_path: str,
    snapshot_dir: str = "./snapshots",
    forecast_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    便捷函数：执行 KGCN 推理

    Args:
        topic_name (str): 话题名称
        model_path (str): 模型权重路径
        config_path (str): 模型配置路径
        snapshot_dir (str): 快照根目录
        forecast_days (Optional[int]): 预测天数，None 则使用模型配置

    Returns:
        Dict[str, Any]: 预测结果字典

    Example:
        >>> result = run_inference(
        ...     topic_name="某热点话题",
        ...     model_path="models/某热点话题/kgcn_xxx.pth",
        ...     config_path="models/某热点话题/kgcn_xxx.json",
        ...     snapshot_dir="./snapshots",
        ...     forecast_days=3,
        ... )
        >>> print(f"风险等级：{result['risk_levels'][0]['level']}")
        >>> for anomaly in result['anomalies']:
        ...     print(f"异常：{anomaly['description']}")
    """
    predictor = KGCNPredictor(
        model_path=model_path,
        config_path=config_path,
        snapshot_dir=snapshot_dir,
    )
    return predictor.predict(topic_name, forecast_days=forecast_days)


def print_inference_report(result: Dict[str, Any]) -> None:
    """
    格式化打印推理报告

    Args:
        result (Dict[str, Any]): predict() 返回的结果字典
    """
    print("\n" + "=" * 70)
    print(f"  📊 舆情预测报告 — 话题：{result['topic_name']}")
    print("=" * 70)
    print(f"  预测时间：{result['predicted_at'][:19]}")
    print(f"  使用历史快照：{result['snapshot_count']} 天")
    print(f"  预测置信度：{result['confidence']:.1%}")

    print("\n  【历史情感趋势】")
    for h in result["history"][-5:]:  # 最近 5 天
        print(
            f"    {h['date']}  正={h['positive']:.2f} 中={h['neutral']:.2f} "
            f"负={h['negative']:.2f}  评论={h['comment_count']}"
        )

    print(f"\n  【未来 {result['forecast_days']} 天预测】")
    for pred, risk in zip(result["predictions"], result["risk_levels"]):
        level_symbol = {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠", 5: "🔴"}.get(risk["level"], "⚪")
        print(
            f"    {pred['date']}  "
            f"正={pred['positive']:.2f} 中={pred['neutral']:.2f} 负={pred['negative']:.2f}  "
            f"{level_symbol} L{risk['level']} {risk['color'].upper()}"
        )

    anomalies = result["anomalies"]
    if anomalies:
        print(f"\n  【异常信号（共 {len(anomalies)} 个）】")
        for a in anomalies[:5]:  # 最多显示 5 个
            severity_bar = "█" * int(a["severity"] * 10) + "░" * (10 - int(a["severity"] * 10))
            print(f"    ⚠️  [{a['type']}] {a['date']}")
            print(f"       {a['description']}")
            print(f"       严重程度：{severity_bar} {a['severity']:.1%}")
    else:
        print("\n  【异常信号】：未检测到异常")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# 可运行示例（完整 Pipeline）
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from kg_snapshot_builder import _generate_demo_csv, build_snapshots
    from kgcn_trainer import train_kgcn

    print("=" * 60)
    print("kgcn_inference.py KGCN 推理与异常检测完整示例")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = os.path.join(tmp_dir, "demo.csv")
        snap_dir = os.path.join(tmp_dir, "snapshots")
        model_dir = os.path.join(tmp_dir, "models")
        topic = "某热点话题"

        # ── 步骤1：准备数据 ────────────────────────────────────────────────
        print("\n[步骤1] 准备演示数据...")
        _generate_demo_csv(csv_file, num_days=14, comments_per_day=20)
        snapshots = build_snapshots(
            csv_path=csv_file,
            topic_name=topic,
            snapshot_dir=snap_dir,
            keep_days=14,
        )
        print(f"  已构建 {len(snapshots)} 个快照")

        # ── 步骤2：训练模型 ────────────────────────────────────────────────
        print("\n[步骤2] 训练 KGCN 模型...")
        train_result = train_kgcn(
            topic_name=topic,
            snapshot_dir=snap_dir,
            model_save_path=model_dir,
            epochs=60,
            batch_size=8,
            forecast_days=3,
        )
        print(f"  训练完成：{train_result['epochs_trained']} 轮，"
              f"最佳验证损失={train_result['best_val_loss']:.4f}")

        # ── 步骤3：加载模型执行推理 ────────────────────────────────────────
        print("\n[步骤3] 执行模型推理...")
        result = run_inference(
            topic_name=topic,
            model_path=train_result["model_path"],
            config_path=train_result["config_path"],
            snapshot_dir=snap_dir,
            forecast_days=3,
        )

        # ── 步骤4：展示预测报告 ────────────────────────────────────────────
        print_inference_report(result)

        # ── 步骤5：验证输出格式 ────────────────────────────────────────────
        print("\n[步骤5] 验证输出格式...")
        assert "predictions" in result, "缺少 predictions 字段"
        assert "risk_levels" in result, "缺少 risk_levels 字段"
        assert "anomalies" in result, "缺少 anomalies 字段"
        assert "confidence" in result, "缺少 confidence 字段"
        assert len(result["predictions"]) == 3, "预测天数不正确"
        for pred in result["predictions"]:
            total = pred["positive"] + pred["neutral"] + pred["negative"]
            assert abs(total - 1.0) < 0.01, f"情感分布不归一（总和={total:.4f}）"
        print("  ✅ 所有字段验证通过")

        # ── 步骤6：演示 load() 类方法 ──────────────────────────────────────
        print("\n[步骤6] 测试 KGCNPredictor.load() 方法...")
        predictor2 = KGCNPredictor.load(
            topic_name=topic,
            model_dir=model_dir,
            snapshot_dir=snap_dir,
        )
        result2 = predictor2.predict(topic)
        print(f"  ✅ 自动加载最新模型并预测成功（置信度={result2['confidence']:.1%}）")

        print("\n✅ KGCN 推理与异常检测全流程验证通过")
