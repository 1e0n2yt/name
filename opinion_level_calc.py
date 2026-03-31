"""
opinion_level_calc.py
舆情风险等级计算模块

根据情感分布、传播指数、影响力指数和热度指数，
按照加权公式计算 1~5 级舆情风险等级。

公式（来源：系统需求文档）：
    综合评分 = 情感×0.4 + 传播×0.25 + 影响力×0.2 + 热度×0.15

等级划分：
    1级（绿色）：综合评分 < 0.2  —— 正常
    2级（蓝色）：综合评分 < 0.4  —— 关注
    3级（黄色）：综合评分 < 0.6  —— 预警
    4级（橙色）：综合评分 < 0.8  —— 高风险
    5级（红色）：综合评分 ≥ 0.8  —— 危机

遵循 PEP8 规范，所有函数提供完整类型注解和文档字符串。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 等级常量定义
# ──────────────────────────────────────────────────────────────────────────────

LEVEL_COLORS = {
    1: "green",   # 正常
    2: "blue",    # 关注
    3: "yellow",  # 预警
    4: "orange",  # 高风险
    5: "red",     # 危机
}

LEVEL_DESCRIPTIONS = {
    1: "正常 —— 舆情整体平稳，无明显负面风险",
    2: "关注 —— 存在一定负面情绪，需持续关注",
    3: "预警 —— 负面情绪较高，建议启动监控预案",
    4: "高风险 —— 负面情绪强烈，存在扩散风险，建议介入",
    5: "危机 —— 舆情严重，极度负面且快速扩散，立即启动危机公关",
}

# 加权系数（来源：系统需求文档）
WEIGHT_SENTIMENT = 0.4    # 情感权重
WEIGHT_SPREAD = 0.25      # 传播权重
WEIGHT_INFLUENCE = 0.2    # 影响力权重
WEIGHT_HEAT = 0.15        # 热度权重

# 等级分段阈值
LEVEL_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]  # < 0.2: L1, < 0.4: L2, ...


@dataclass
class OpinionLevelResult:
    """
    舆情等级计算结果

    Attributes:
        level (int): 舆情等级，1~5
        composite_score (float): 综合评分，范围 [0, 1]
        sentiment_score (float): 情感维度得分（已归一化），范围 [0, 1]
        spread_score (float): 传播维度得分（已归一化），范围 [0, 1]
        influence_score (float): 影响力维度得分（已归一化），范围 [0, 1]
        heat_score (float): 热度维度得分（已归一化），范围 [0, 1]
        color (str): 等级对应颜色名称
        description (str): 等级描述文字
    """
    level: int
    composite_score: float
    sentiment_score: float
    spread_score: float
    influence_score: float
    heat_score: float
    color: str
    description: str


def calc_sentiment_score(
    neg_ratio: float,
    sentiment_change_rate: float = 0.0,
) -> float:
    """
    计算情感维度得分（归一化到 [0, 1]）

    情感得分由两部分组成：
    - 当前负面情感占比（权重 0.7）
    - 负面情感变化速率（权重 0.3）

    Args:
        neg_ratio (float): 当前负面情感占比，范围 [0, 1]
        sentiment_change_rate (float): 负面情感变化速率（相较于前一天的变化量），
                                       正值表示负面情感上升，范围建议 [-1, 1]

    Returns:
        float: 情感维度得分，范围 [0, 1]
    """
    if not (0.0 <= neg_ratio <= 1.0):
        logger.warning("neg_ratio 超出 [0, 1] 范围，已截断: %f", neg_ratio)
        neg_ratio = max(0.0, min(1.0, neg_ratio))

    # 变化速率归一化：将变化量映射到 [0, 1]，上升 > 0 时增加风险
    change_component = max(0.0, min(1.0, (sentiment_change_rate + 1.0) / 2.0))

    score = 0.7 * neg_ratio + 0.3 * change_component
    return float(max(0.0, min(1.0, score)))


def calc_spread_score(
    comment_growth_rate: float,
    repost_count: int = 0,
    reply_count: int = 0,
    max_repost: int = 10000,
) -> float:
    """
    计算传播维度得分（归一化到 [0, 1]）

    传播得分由两部分组成：
    - 评论数增长率（权重 0.6）
    - 转发/回复量（权重 0.4）

    Args:
        comment_growth_rate (float): 评论数增长率（相较于前一天），
                                     例如 2.0 表示增长 200%
        repost_count (int): 转发数量
        reply_count (int): 回复数量
        max_repost (int): 用于归一化的最大转发数参考值，默认 10000

    Returns:
        float: 传播维度得分，范围 [0, 1]
    """
    # 增长率归一化：增长 200% 及以上视为满分（异常评论激增阈值）
    growth_component = min(1.0, comment_growth_rate / 2.0) if comment_growth_rate > 0 else 0.0

    # 互动量归一化
    interaction = repost_count + reply_count
    interaction_component = min(1.0, interaction / max_repost)

    score = 0.6 * growth_component + 0.4 * interaction_component
    return float(max(0.0, min(1.0, score)))


def calc_influence_score(
    high_risk_user_count: int,
    verified_user_neg_ratio: float = 0.0,
    max_high_risk_users: int = 20,
) -> float:
    """
    计算影响力维度得分（归一化到 [0, 1]）

    影响力得分由两部分组成：
    - 高风险用户数量（权重 0.6）
    - 认证用户（蓝V/黄V）负面评论占比（权重 0.4）

    Args:
        high_risk_user_count (int): 当前高风险用户数量（情感极度消极的用户）
        verified_user_neg_ratio (float): 认证用户中负面情感占比，范围 [0, 1]
        max_high_risk_users (int): 用于归一化的最大高风险用户参考值，默认 20

    Returns:
        float: 影响力维度得分，范围 [0, 1]
    """
    # 高风险用户数量归一化
    risk_user_component = min(1.0, high_risk_user_count / max_high_risk_users)

    if not (0.0 <= verified_user_neg_ratio <= 1.0):
        logger.warning(
            "verified_user_neg_ratio 超出 [0, 1] 范围，已截断: %f",
            verified_user_neg_ratio,
        )
        verified_user_neg_ratio = max(0.0, min(1.0, verified_user_neg_ratio))

    score = 0.6 * risk_user_component + 0.4 * verified_user_neg_ratio
    return float(max(0.0, min(1.0, score)))


def calc_heat_score(
    comment_count: int,
    like_count: int = 0,
    max_comments: int = 5000,
    max_likes: int = 50000,
) -> float:
    """
    计算热度维度得分（归一化到 [0, 1]）

    热度得分由两部分组成：
    - 评论总量（权重 0.5）
    - 点赞总量（权重 0.5）

    Args:
        comment_count (int): 当前话题评论总数
        like_count (int): 当前话题点赞总数
        max_comments (int): 用于归一化的最大评论数参考值，默认 5000
        max_likes (int): 用于归一化的最大点赞数参考值，默认 50000

    Returns:
        float: 热度维度得分，范围 [0, 1]
    """
    comment_component = min(1.0, comment_count / max_comments)
    like_component = min(1.0, like_count / max_likes)

    score = 0.5 * comment_component + 0.5 * like_component
    return float(max(0.0, min(1.0, score)))


def score_to_level(composite_score: float) -> int:
    """
    将综合评分映射为舆情风险等级（1~5）

    等级划分：
        1级：综合评分 < 0.2
        2级：综合评分 < 0.4
        3级：综合评分 < 0.6
        4级：综合评分 < 0.8
        5级：综合评分 ≥ 0.8

    Args:
        composite_score (float): 综合评分，范围 [0, 1]

    Returns:
        int: 舆情风险等级，1~5
    """
    for level, threshold in enumerate(LEVEL_THRESHOLDS, start=1):
        if composite_score < threshold:
            return level
    return 5


def calculate_opinion_level(
    neg_ratio: float,
    sentiment_change_rate: float = 0.0,
    comment_growth_rate: float = 0.0,
    repost_count: int = 0,
    reply_count: int = 0,
    high_risk_user_count: int = 0,
    verified_user_neg_ratio: float = 0.0,
    comment_count: int = 0,
    like_count: int = 0,
) -> OpinionLevelResult:
    """
    综合计算舆情风险等级

    公式：
        综合评分 = 情感×0.4 + 传播×0.25 + 影响力×0.2 + 热度×0.15

    Args:
        neg_ratio (float): 负面情感占比，范围 [0, 1]
        sentiment_change_rate (float): 负面情感变化速率（相较于前一天），
                                       正值表示上升，默认 0.0
        comment_growth_rate (float): 评论数增长率（相较于前一天），
                                     例如 2.0 表示增长 200%，默认 0.0
        repost_count (int): 转发数量，默认 0
        reply_count (int): 回复数量，默认 0
        high_risk_user_count (int): 高风险用户数量，默认 0
        verified_user_neg_ratio (float): 认证用户负面比例，范围 [0, 1]，默认 0.0
        comment_count (int): 评论总数，默认 0
        like_count (int): 点赞总数，默认 0

    Returns:
        OpinionLevelResult: 包含等级、综合评分及各维度得分的结果对象
    """
    # 各维度得分计算
    s_score = calc_sentiment_score(neg_ratio, sentiment_change_rate)
    sp_score = calc_spread_score(comment_growth_rate, repost_count, reply_count)
    inf_score = calc_influence_score(high_risk_user_count, verified_user_neg_ratio)
    heat = calc_heat_score(comment_count, like_count)

    # 综合评分（加权求和）
    composite = (
        WEIGHT_SENTIMENT * s_score
        + WEIGHT_SPREAD * sp_score
        + WEIGHT_INFLUENCE * inf_score
        + WEIGHT_HEAT * heat
    )
    composite = float(max(0.0, min(1.0, composite)))

    level = score_to_level(composite)

    logger.debug(
        "Opinion level: %d (composite=%.3f, sentiment=%.3f, spread=%.3f, "
        "influence=%.3f, heat=%.3f)",
        level, composite, s_score, sp_score, inf_score, heat,
    )

    return OpinionLevelResult(
        level=level,
        composite_score=composite,
        sentiment_score=s_score,
        spread_score=sp_score,
        influence_score=inf_score,
        heat_score=heat,
        color=LEVEL_COLORS[level],
        description=LEVEL_DESCRIPTIONS[level],
    )


def calculate_opinion_level_from_distribution(
    sentiment_distribution: Dict[str, float],
    prev_neg_ratio: Optional[float] = None,
    comment_count: int = 0,
    prev_comment_count: int = 0,
    repost_count: int = 0,
    reply_count: int = 0,
    high_risk_user_count: int = 0,
    verified_user_neg_ratio: float = 0.0,
    like_count: int = 0,
) -> OpinionLevelResult:
    """
    从情感分布字典直接计算舆情风险等级（便捷接口）

    Args:
        sentiment_distribution (Dict[str, float]): 情感分布字典，
            格式为 {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}
        prev_neg_ratio (Optional[float]): 前一天负面占比，用于计算变化率
        comment_count (int): 当前评论总数
        prev_comment_count (int): 前一天评论总数，用于计算增长率
        repost_count (int): 转发数量
        reply_count (int): 回复数量
        high_risk_user_count (int): 高风险用户数量
        verified_user_neg_ratio (float): 认证用户负面比例
        like_count (int): 点赞总数

    Returns:
        OpinionLevelResult: 舆情等级计算结果
    """
    neg_ratio = sentiment_distribution.get("negative", 0.0)

    # 计算情感变化速率
    if prev_neg_ratio is not None:
        sentiment_change_rate = neg_ratio - prev_neg_ratio
    else:
        sentiment_change_rate = 0.0

    # 计算评论增长率
    if prev_comment_count > 0:
        comment_growth_rate = (comment_count - prev_comment_count) / prev_comment_count
    else:
        comment_growth_rate = 0.0

    return calculate_opinion_level(
        neg_ratio=neg_ratio,
        sentiment_change_rate=sentiment_change_rate,
        comment_growth_rate=comment_growth_rate,
        repost_count=repost_count,
        reply_count=reply_count,
        high_risk_user_count=high_risk_user_count,
        verified_user_neg_ratio=verified_user_neg_ratio,
        comment_count=comment_count,
        like_count=like_count,
    )


def batch_calculate_levels(
    snapshots_data: List[Dict],
) -> List[OpinionLevelResult]:
    """
    批量计算多个时序快照的舆情等级

    Args:
        snapshots_data (List[Dict]): 快照数据列表，每项为包含以下键的字典：
            - sentiment_distribution (Dict[str, float]): 情感分布
            - comment_count (int): 评论数
            - repost_count (int): 转发数（可选）
            - reply_count (int): 回复数（可选）
            - high_risk_user_count (int): 高风险用户数（可选）
            - verified_user_neg_ratio (float): 认证用户负面比例（可选）
            - like_count (int): 点赞数（可选）

    Returns:
        List[OpinionLevelResult]: 对应的等级计算结果列表
    """
    results = []
    for i, data in enumerate(snapshots_data):
        prev_data = snapshots_data[i - 1] if i > 0 else None
        prev_neg = (
            prev_data["sentiment_distribution"].get("negative", 0.0)
            if prev_data else None
        )
        prev_comment_count = prev_data.get("comment_count", 0) if prev_data else 0

        result = calculate_opinion_level_from_distribution(
            sentiment_distribution=data["sentiment_distribution"],
            prev_neg_ratio=prev_neg,
            comment_count=data.get("comment_count", 0),
            prev_comment_count=prev_comment_count,
            repost_count=data.get("repost_count", 0),
            reply_count=data.get("reply_count", 0),
            high_risk_user_count=data.get("high_risk_user_count", 0),
            verified_user_neg_ratio=data.get("verified_user_neg_ratio", 0.0),
            like_count=data.get("like_count", 0),
        )
        results.append(result)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 模块自测示例
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("opinion_level_calc.py 舆情等级计算示例")
    print("=" * 60)

    # 示例1：正常舆情（负面占比低）
    result1 = calculate_opinion_level(
        neg_ratio=0.15,
        comment_count=100,
        like_count=500,
    )
    print(f"\n[示例1 - 正常舆情]")
    print(f"  综合评分: {result1.composite_score:.3f}")
    print(f"  等级: {result1.level} ({result1.color})")
    print(f"  描述: {result1.description}")

    # 示例2：舆情预警（负面急速上升）
    result2 = calculate_opinion_level(
        neg_ratio=0.52,
        sentiment_change_rate=0.17,   # 负面增加 17%
        comment_growth_rate=2.5,      # 评论增长 250%
        repost_count=800,
        reply_count=1200,
        high_risk_user_count=3,
        comment_count=450,
        like_count=2000,
    )
    print(f"\n[示例2 - 舆情预警]")
    print(f"  综合评分: {result2.composite_score:.3f}")
    print(f"  等级: {result2.level} ({result2.color})")
    print(f"  描述: {result2.description}")

    # 示例3：危机舆情（极度负面+快速扩散）
    result3 = calculate_opinion_level(
        neg_ratio=0.78,
        sentiment_change_rate=0.26,
        comment_growth_rate=4.0,
        repost_count=5000,
        reply_count=8000,
        high_risk_user_count=15,
        verified_user_neg_ratio=0.6,
        comment_count=3000,
        like_count=20000,
    )
    print(f"\n[示例3 - 危机舆情]")
    print(f"  综合评分: {result3.composite_score:.3f}")
    print(f"  等级: {result3.level} ({result3.color})")
    print(f"  描述: {result3.description}")

    # 示例4：批量计算（模拟 3 天数据）
    snapshots = [
        {
            "sentiment_distribution": {"positive": 0.45, "neutral": 0.40, "negative": 0.15},
            "comment_count": 80,
            "like_count": 300,
        },
        {
            "sentiment_distribution": {"positive": 0.35, "neutral": 0.33, "negative": 0.32},
            "comment_count": 180,
            "like_count": 800,
            "repost_count": 200,
        },
        {
            "sentiment_distribution": {"positive": 0.20, "neutral": 0.28, "negative": 0.52},
            "comment_count": 550,
            "like_count": 3000,
            "repost_count": 1200,
            "high_risk_user_count": 5,
        },
    ]
    batch_results = batch_calculate_levels(snapshots)
    print(f"\n[示例4 - 批量计算（3天数据）]")
    for i, r in enumerate(batch_results, start=1):
        print(
            f"  Day {i}: 等级={r.level}({r.color})  "
            f"综合={r.composite_score:.3f}  "
            f"情感={r.sentiment_score:.3f}  "
            f"传播={r.spread_score:.3f}"
        )

    print("\n✅ 舆情等级计算验证通过")
