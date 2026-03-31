"""
social_media_models.py
社交媒体数据模型定义模块

定义舆情监控系统中使用的核心数据模型：
- User: 微博用户模型
- Post: 微博帖子模型
- Comment: 评论模型
- Keyword: 关键词模型
- SentimentLabel: 情感标签模型

遵循 PEP8 规范，所有字段严格按照系统规范定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class User:
    """
    微博用户模型

    Attributes:
        user_id (str): 用户唯一标识符
        created_at (datetime): 账号创建时间
        followers (int): 粉丝数量
        following (int): 关注数量
        verified (bool): 是否为认证用户（蓝V/黄V）
        total_posts (int): 历史总发帖数
        engagement_rate (float): 互动率（互动量 / 曝光量），范围 [0, 1]
    """
    user_id: str
    created_at: datetime
    followers: int
    following: int
    verified: bool
    total_posts: int
    engagement_rate: float

    def __post_init__(self):
        """数据校验"""
        if self.followers < 0:
            raise ValueError(f"followers 不能为负数，当前值：{self.followers}")
        if self.following < 0:
            raise ValueError(f"following 不能为负数，当前值：{self.following}")
        if self.total_posts < 0:
            raise ValueError(f"total_posts 不能为负数，当前值：{self.total_posts}")
        if not (0.0 <= self.engagement_rate <= 1.0):
            raise ValueError(
                f"engagement_rate 必须在 [0, 1] 范围内，当前值：{self.engagement_rate}"
            )

    def is_influential(self, followers_threshold: int = 1000) -> bool:
        """
        判断是否为影响力用户

        Args:
            followers_threshold (int): 粉丝数阈值，默认 1000

        Returns:
            bool: 是否为影响力用户
        """
        return self.followers >= followers_threshold or self.verified


@dataclass
class Post:
    """
    微博帖子模型

    Attributes:
        post_id (str): 帖子唯一标识符
        user_id (str): 发帖用户 ID
        content_clean (str): 清洗后的帖子内容文本
        created_at (datetime): 帖子发布时间
        likes (int): 点赞数
        replies (int): 回复数
        reposts (int): 转发数
        viral_coefficient (float): 病毒传播系数，衡量内容传播能力，范围 [0, +∞)
        sentiment_score (float): 情感得分，范围 [-1.0, 1.0]，
                                 -1 表示极度消极，0 表示中性，1 表示极度积极
        keywords (List[str]): 帖子关键词列表
    """
    post_id: str
    user_id: str
    content_clean: str
    created_at: datetime
    likes: int
    replies: int
    reposts: int
    viral_coefficient: float
    sentiment_score: float
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        """数据校验"""
        if self.likes < 0:
            raise ValueError(f"likes 不能为负数，当前值：{self.likes}")
        if self.replies < 0:
            raise ValueError(f"replies 不能为负数，当前值：{self.replies}")
        if self.reposts < 0:
            raise ValueError(f"reposts 不能为负数，当前值：{self.reposts}")
        if self.viral_coefficient < 0:
            raise ValueError(
                f"viral_coefficient 不能为负数，当前值：{self.viral_coefficient}"
            )
        if not (-1.0 <= self.sentiment_score <= 1.0):
            raise ValueError(
                f"sentiment_score 必须在 [-1, 1] 范围内，当前值：{self.sentiment_score}"
            )

    def get_sentiment_label(self) -> str:
        """
        根据情感分数返回情感标签

        Returns:
            str: 'positive'（积极）| 'neutral'（中性）| 'negative'（消极）
        """
        if self.sentiment_score > 0.1:
            return "positive"
        elif self.sentiment_score < -0.1:
            return "negative"
        else:
            return "neutral"

    def get_engagement_count(self) -> int:
        """
        获取总互动量（点赞 + 回复 + 转发）

        Returns:
            int: 总互动量
        """
        return self.likes + self.replies + self.reposts


@dataclass
class Comment:
    """
    评论模型

    Attributes:
        comment_id (str): 评论唯一标识符
        user_id (str): 评论用户 ID
        post_id (str): 所属帖子 ID
        content_clean (str): 清洗后的评论内容文本
        created_at (datetime): 评论发布时间
        likes (int): 点赞数
        replies (int): 回复数
        sentiment_score (float): 情感得分，范围 [-1.0, 1.0]，
                                 -1 表示极度消极，0 表示中性，1 表示极度积极
        keywords (List[str]): 评论关键词列表
    """
    comment_id: str
    user_id: str
    post_id: str
    content_clean: str
    created_at: datetime
    likes: int
    replies: int
    sentiment_score: float
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        """数据校验"""
        if self.likes < 0:
            raise ValueError(f"likes 不能为负数，当前值：{self.likes}")
        if self.replies < 0:
            raise ValueError(f"replies 不能为负数，当前值：{self.replies}")
        if not (-1.0 <= self.sentiment_score <= 1.0):
            raise ValueError(
                f"sentiment_score 必须在 [-1, 1] 范围内，当前值：{self.sentiment_score}"
            )

    def get_sentiment_label(self) -> str:
        """
        根据情感分数返回情感标签

        Returns:
            str: 'positive'（积极）| 'neutral'（中性）| 'negative'（消极）
        """
        if self.sentiment_score > 0.1:
            return "positive"
        elif self.sentiment_score < -0.1:
            return "negative"
        else:
            return "neutral"

    def is_high_risk(self, neg_threshold: float = -0.6) -> bool:
        """
        判断是否为高风险评论（情感极度消极）

        Args:
            neg_threshold (float): 消极情感阈值，默认 -0.6

        Returns:
            bool: 是否为高风险评论
        """
        return self.sentiment_score <= neg_threshold


@dataclass
class Keyword:
    """
    关键词模型

    Attributes:
        keyword (str): 关键词文本
        frequency (int): 在文档集合中出现的总频次
        tf_idf (float): TF-IDF 权重值，衡量关键词重要程度
        positive_context_count (int): 出现在积极情感语境中的次数
        negative_context_count (int): 出现在消极情感语境中的次数
    """
    keyword: str
    frequency: int
    tf_idf: float
    positive_context_count: int
    negative_context_count: int

    def __post_init__(self):
        """数据校验"""
        if self.frequency < 0:
            raise ValueError(f"frequency 不能为负数，当前值：{self.frequency}")
        if self.tf_idf < 0:
            raise ValueError(f"tf_idf 不能为负数，当前值：{self.tf_idf}")
        if self.positive_context_count < 0:
            raise ValueError(
                f"positive_context_count 不能为负数，当前值：{self.positive_context_count}"
            )
        if self.negative_context_count < 0:
            raise ValueError(
                f"negative_context_count 不能为负数，当前值：{self.negative_context_count}"
            )

    def get_sentiment_tendency(self) -> str:
        """
        计算关键词的情感倾向

        Returns:
            str: 'positive'（积极倾向）| 'neutral'（中性）| 'negative'（消极倾向）
        """
        total = self.positive_context_count + self.negative_context_count
        if total == 0:
            return "neutral"
        pos_ratio = self.positive_context_count / total
        if pos_ratio > 0.6:
            return "positive"
        elif pos_ratio < 0.4:
            return "negative"
        else:
            return "neutral"

    def get_controversy_score(self) -> float:
        """
        计算争议性分数（正负情感出现比例接近 0.5 时争议性最高）

        Returns:
            float: 争议性分数，范围 [0, 1]，越接近 1 表示越有争议
        """
        total = self.positive_context_count + self.negative_context_count
        if total == 0:
            return 0.0
        pos_ratio = self.positive_context_count / total
        # 使用 1 - |2*p - 1| 计算争议性：p=0.5 时争议性为 1，p=0 或 1 时为 0
        return 1.0 - abs(2 * pos_ratio - 1)


@dataclass
class SentimentLabel:
    """
    情感标签模型，用于知识图谱中的情感节点

    Attributes:
        label (str): 情感标签，取值 'positive'（积极）| 'neutral'（中性）| 'negative'（消极）
        score_range_min (float): 该标签对应的情感分数最小值
        score_range_max (float): 该标签对应的情感分数最大值
    """
    label: str
    score_range_min: float
    score_range_max: float

    # 预定义的三种情感标签
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

    VALID_LABELS = {POSITIVE, NEUTRAL, NEGATIVE}

    def __post_init__(self):
        """数据校验"""
        if self.label not in self.VALID_LABELS:
            raise ValueError(
                f"label 必须为 {self.VALID_LABELS} 中的一个，当前值：{self.label}"
            )
        if self.score_range_min > self.score_range_max:
            raise ValueError(
                f"score_range_min ({self.score_range_min}) "
                f"不能大于 score_range_max ({self.score_range_max})"
            )

    @classmethod
    def create_positive(cls) -> "SentimentLabel":
        """创建积极情感标签实例"""
        return cls(label=cls.POSITIVE, score_range_min=0.1, score_range_max=1.0)

    @classmethod
    def create_neutral(cls) -> "SentimentLabel":
        """创建中性情感标签实例"""
        return cls(label=cls.NEUTRAL, score_range_min=-0.1, score_range_max=0.1)

    @classmethod
    def create_negative(cls) -> "SentimentLabel":
        """创建消极情感标签实例"""
        return cls(label=cls.NEGATIVE, score_range_min=-1.0, score_range_max=-0.1)

    @classmethod
    def from_score(cls, score: float) -> "SentimentLabel":
        """
        根据情感分数创建对应的情感标签实例

        Args:
            score (float): 情感得分，范围 [-1.0, 1.0]

        Returns:
            SentimentLabel: 对应的情感标签实例
        """
        if score > 0.1:
            return cls.create_positive()
        elif score < -0.1:
            return cls.create_negative()
        else:
            return cls.create_neutral()


# ──────────────────────────────────────────────────────────────────────────────
# 模块自测示例
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("social_media_models.py 模型验证示例")
    print("=" * 60)

    # 创建用户示例
    user = User(
        user_id="u_001",
        created_at=datetime(2020, 1, 1),
        followers=5000,
        following=200,
        verified=True,
        total_posts=320,
        engagement_rate=0.045,
    )
    print(f"\n[User] {user.user_id}")
    print(f"  粉丝数: {user.followers}, 认证: {user.verified}")
    print(f"  是否影响力用户: {user.is_influential()}")

    # 创建帖子示例
    post = Post(
        post_id="p_001",
        user_id="u_001",
        content_clean="这个产品质量真的很差，太失望了！",
        created_at=datetime(2024, 3, 1, 10, 0, 0),
        likes=120,
        replies=45,
        reposts=30,
        viral_coefficient=1.8,
        sentiment_score=-0.75,
        keywords=["质量", "失望"],
    )
    print(f"\n[Post] {post.post_id}")
    print(f"  情感分数: {post.sentiment_score}")
    print(f"  情感标签: {post.get_sentiment_label()}")
    print(f"  总互动量: {post.get_engagement_count()}")

    # 创建评论示例
    comment = Comment(
        comment_id="c_001",
        user_id="u_002",
        post_id="p_001",
        content_clean="完全同意，买了后悔！",
        created_at=datetime(2024, 3, 1, 11, 0, 0),
        likes=50,
        replies=5,
        sentiment_score=-0.8,
        keywords=["后悔"],
    )
    print(f"\n[Comment] {comment.comment_id}")
    print(f"  情感分数: {comment.sentiment_score}")
    print(f"  情感标签: {comment.get_sentiment_label()}")
    print(f"  是否高风险: {comment.is_high_risk()}")

    # 创建关键词示例
    keyword = Keyword(
        keyword="质量问题",
        frequency=85,
        tf_idf=0.312,
        positive_context_count=10,
        negative_context_count=75,
    )
    print(f"\n[Keyword] {keyword.keyword}")
    print(f"  TF-IDF: {keyword.tf_idf:.3f}")
    print(f"  情感倾向: {keyword.get_sentiment_tendency()}")
    print(f"  争议性分数: {keyword.get_controversy_score():.3f}")

    # 创建情感标签示例
    sentiment_neg = SentimentLabel.from_score(-0.8)
    print(f"\n[SentimentLabel] from_score(-0.8) → label: {sentiment_neg.label}")

    print("\n✅ 所有模型验证通过")
