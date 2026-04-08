"""
data_models.py - 舆情监控系统共享数据模型

设计改进说明：
  1. 粉丝量/点赞量/回复量 → 改为节点属性，不再是独立实体
  2. 情感分数使用连续 [0, 1] 区间（0=极端负面, 1=极端正面）
  3. 移除 sentiment_label（由分数区间推导），Sentiment 节点保留备用
  4. 新增 Keyword 节点（词频、TF-IDF、情感倾向）
  5. 新增 TimeSlot 节点（聚合时间信息）
  6. 新增 Community 节点（非连通图检测）
  7. 完整的 User / Post / Comment 属性系统
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# 情感分数区间定义 [0, 1]
# ---------------------------------------------------------------------------
class SentimentRange:
    """
    情感分数区间标准（用分数区间判断类别，不保存离散标签）

    [0.00, 0.20)  → 极端负面
    [0.20, 0.40)  → 负面
    [0.40, 0.60)  → 中性
    [0.60, 0.80)  → 正面
    [0.80, 1.00]  → 极端正面
    """
    EXTREME_NEGATIVE_MAX = 0.20
    NEGATIVE_MAX = 0.40
    NEUTRAL_MAX = 0.60
    POSITIVE_MAX = 0.80
    # >= 0.80 → 极端正面

    @staticmethod
    def label(score: float) -> str:
        """根据分数返回人类可读的情感类别（仅供展示，不持久化）。"""
        if score < SentimentRange.EXTREME_NEGATIVE_MAX:
            return "extreme_negative"
        if score < SentimentRange.NEGATIVE_MAX:
            return "negative"
        if score < SentimentRange.NEUTRAL_MAX:
            return "neutral"
        if score < SentimentRange.POSITIVE_MAX:
            return "positive"
        return "extreme_positive"

    @staticmethod
    def is_negative(score: float) -> bool:
        return score < SentimentRange.NEUTRAL_MAX

    @staticmethod
    def is_extreme(score: float) -> bool:
        return score < SentimentRange.EXTREME_NEGATIVE_MAX or score >= SentimentRange.POSITIVE_MAX


# ---------------------------------------------------------------------------
# 节点类型和关系类型常量
# ---------------------------------------------------------------------------
class NodeType:
    USER = "user"
    POST = "post"            # 原始微博/博文
    COMMENT = "comment"      # 评论
    KEYWORD = "keyword"      # 关键词节点
    TOPIC = "topic"          # 话题标签 #xxx#
    ENTITY = "entity"        # 命名实体（人物/品牌/事件）
    TIME_SLOT = "time_slot"  # 时间槽（小时或天级别）
    COMMUNITY = "community"  # 社群节点（用于非连通图检测）


class RelationType:
    # ---- 用户行为 ----
    USER_WRITES_POST = "writes_post"          # 用户 → 发布 → 博文
    USER_WRITES_COMMENT = "writes_comment"    # 用户 → 发表 → 评论
    USER_LIKES_POST = "likes_post"            # 用户 → 点赞 → 博文
    USER_REPOSTS = "reposts"                  # 用户 → 转发 → 博文
    USER_FOLLOWS = "follows"                  # 用户 → 关注 → 用户

    # ---- 评论/博文关系 ----
    COMMENT_REPLIES_TO = "replies_to"         # 评论 → 回复 → 评论/博文
    POST_CONTAINS_KEYWORD = "post_has_kw"     # 博文 → 包含 → 关键词
    COMMENT_CONTAINS_KEYWORD = "comment_has_kw"  # 评论 → 包含 → 关键词
    POST_MENTIONS_TOPIC = "mentions_topic"    # 博文/评论 → 提及 → 话题
    POST_MENTIONS_ENTITY = "mentions_entity"  # 博文/评论 → 提及 → 实体

    # ---- 时间关系 ----
    AT_TIME_SLOT = "at_time_slot"             # 节点 → 发生于 → 时间槽
    FOLLOWS_IN_TIME = "follows_in_time"       # 评论1 → 时序后 → 评论2

    # ---- 用户情感轨迹 ----
    USER_SENTIMENT_TRAJECTORY = "sentiment_trajectory"  # 同用户评论间的情感演变

    # ---- 社群关系 ----
    BELONGS_TO_COMMUNITY = "belongs_to_community"  # 节点 → 属于 → 社群


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class User:
    """
    用户节点属性

    改进说明：粉丝量/认证状态/账龄/信誉分作为属性保存，
              不再作为独立节点存在。
    """
    user_id: str
    username: str

    # 影响力属性
    followers_count: int = 0          # 粉丝数（原"粉丝量"节点 → 属性）
    following_count: int = 0          # 关注数
    is_certified: bool = False        # 认证用户标志
    certification_type: str = ""      # 认证类型："media"/"government"/"celebrity"/""

    # 活跃度属性
    account_age_days: int = 0         # 账龄（天）
    total_posts: int = 0              # 历史发帖总数
    avg_engagement: float = 0.0       # 平均互动率（(点赞+转发+评论)/粉丝数）

    # 信誉与影响
    reputation_score: float = 0.5    # 信誉分 [0, 1]，越高越可信
    influence_score: float = 0.0     # 影响力分数（由 followers / 认证状态计算）

    # 时间信息
    registered_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None

    def compute_influence_score(self) -> float:
        """
        计算影响力分数 [0, 1]：
          基础 = log10(followers + 1) / 7  （上限约 1000 万粉）
          认证加成：+0.2
          账龄加成：min(0.1, account_age_days / 3650 * 0.1)
        """
        import math
        base = min(1.0, math.log10(self.followers_count + 1) / 7.0)
        cert_bonus = 0.2 if self.is_certified else 0.0
        age_bonus = min(0.1, self.account_age_days / 3650.0 * 0.1)
        self.influence_score = min(1.0, base + cert_bonus + age_bonus)
        return self.influence_score


@dataclass
class Post:
    """
    博文（原始微博）节点属性

    改进说明：点赞量/转发量/回复量作为属性，不再是独立节点。
    """
    post_id: str
    author_id: str
    content: str
    content_clean: str = ""

    # 互动数据（原"点赞量"/"回复量"节点 → 属性）
    likes_count: int = 0              # 点赞数
    reposts_count: int = 0            # 转发数
    comments_count: int = 0          # 评论数

    # 传播属性
    spread_coefficient: float = 0.0  # 传播系数 = reposts / (likes + 1)
    virality_score: float = 0.0      # 病毒传播分数 [0, 1]

    # 情感信息（连续分数，不保存离散标签）
    sentiment_score: float = 0.5     # [0, 1]，0=极端负面，1=极端正面

    # 内容特征
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)

    # 时间信息
    created_at: Optional[datetime] = None
    time_slot_id: str = ""           # 关联的 TimeSlot 节点 ID

    def compute_virality_score(self) -> float:
        """
        传播系数 [0, 1]：
          raw = reposts * 2 + likes + comments * 1.5
          归一化到 [0, 1]（上限假设 100k 互动）
        """
        raw = self.reposts_count * 2 + self.likes_count + self.comments_count * 1.5
        self.virality_score = min(1.0, raw / 100_000.0)
        self.spread_coefficient = (
            self.reposts_count / (self.likes_count + 1)
        )
        return self.virality_score


@dataclass
class Comment:
    """
    评论节点属性

    改进说明：点赞量/回复量作为属性；情感使用连续分数 [0, 1]。
    """
    comment_id: str
    author_id: str
    post_id: str
    content: str
    content_clean: str = ""

    # 互动数据（原节点 → 属性）
    likes_count: int = 0
    replies_count: int = 0           # 该评论被回复次数

    # 情感信息（连续分数）
    sentiment_score: float = 0.5     # [0, 1]，0=极端负面，1=极端正面

    # 内容特征
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    # 时间信息
    created_at: Optional[datetime] = None
    time_slot_id: str = ""

    # 回复链信息
    parent_comment_id: Optional[str] = None  # 被回复的评论 ID（若有）


@dataclass
class Keyword:
    """
    关键词节点属性

    改进说明（问题4）：添加词频、TF-IDF、情感倾向作为节点属性。
    """
    keyword_id: str                  # 通常就是关键词本身
    word: str

    # 统计特征
    frequency: int = 0               # 在当前时间窗口内的总出现次数
    tfidf_score: float = 0.0         # TF-IDF 分数（归一化）
    document_count: int = 0          # 出现该词的文档（评论/博文）数量

    # 情感倾向
    sentiment_tendency: float = 0.5  # 该词在上下文中的平均情感分数 [0, 1]
    # 可选：情感极性强度（deviation from neutral）
    sentiment_polarity: float = 0.0  # abs(sentiment_tendency - 0.5) * 2  ∈ [0, 1]

    def update_sentiment_polarity(self):
        self.sentiment_polarity = abs(self.sentiment_tendency - 0.5) * 2.0


@dataclass
class TimeSlot:
    """
    时间槽节点（问题2）

    用于聚合时间信息，支持时序图快照的生成。
    粒度可选：hour / day / week
    """
    slot_id: str                     # 例如 "2024-03-15_14" 或 "2024-03-15"
    start_time: datetime
    end_time: datetime
    granularity: str = "day"         # "hour" | "day" | "week"

    # 聚合统计（在快照生成时填充）
    total_comments: int = 0
    total_users: int = 0
    avg_sentiment: float = 0.5       # [0, 1]
    negative_ratio: float = 0.0      # [0, 1]
    extreme_negative_ratio: float = 0.0


@dataclass
class Community:
    """
    社群节点（问题6）

    用于非连通图的社群检测：每个连通分量或社区算法
    （如 Louvain）识别的聚类对应一个 Community 节点。

    两种设计方案说明：
      方案A（本方案）：Community 作为独立节点，
                      通过 BELONGS_TO_COMMUNITY 边与 User/Post/Comment 相连。
                      优点：易于图查询和遍历；支持跨社群关系分析。
      方案B（替代）：将 community_id 作为 User/Post/Comment 的属性。
                      优点：更简单；缺点：无法在图上直接遍历社群结构。

    非连通图检测说明：
      当图中存在孤立子图时（如两个互不相关的话题），
      各子图会分别被识别为不同的 Community，
      community_size 较小的往往是噪声或边缘话题。
    """
    community_id: str
    label: str = ""                  # 社群的人工标签（如"反对派"/"支持派"）

    # 图拓扑属性
    community_size: int = 0          # 社群内节点数
    internal_edges: int = 0          # 社群内部边数
    external_edges: int = 0          # 与其他社群的连接边数
    density: float = 0.0             # 内部密度 = internal_edges / (size*(size-1))

    # 情感属性
    avg_sentiment: float = 0.5       # 社群平均情感 [0, 1]
    sentiment_homogeneity: float = 0.0  # 情感一致性（越高越极化）

    # 影响力
    total_followers: int = 0         # 社群内用户总粉丝数
    has_kol: bool = False            # 是否包含认证/高粉用户（KOL）

    # 发现时间
    detected_at: Optional[datetime] = None
