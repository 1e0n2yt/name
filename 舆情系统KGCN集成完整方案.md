# 舆情监控系统中的 KGCN 情感预测实现方案

## 概述

本文档提供一套**完整的实践方案**，将 KGCN 集成到你的舆情监控系统中，用于**对热点评论进行情感趋势预测和风险预警**。

> **系统背景**：
> - 已有 BERT 情感分析模型（能对单条文本打分）
> - 每 30 分钟爬取热点微博及评论
> - 需要实现"预防舆情二次发酵"的目标

---

## 第一部分：系统架构调整

### 1.1 现有系统回顾

```
爬虫采集 → 情感分析 → 词云/统计 → 可视化展示
(30min)    (BERT)     (热词)     (Web)
```

你现在有：
- ✅ 文本爬取和清洗
- ✅ 单条文本的情感分析（BERT）
- ❌ 缺少：时序情感预测和趋势识别

### 1.2 新增 KGCN 预测模块

```
┌─────────────────────────────────────────┐
│  爬虫采集评论                            │
│  (热点话题的用户评论序列)                  │
└────────────────┬────────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│  Step 1: 特征提取                         │
│  ├─ BERT 情感分析                         │
│  ├─ 提取评论的关键词和实体                   │
│  └─ 记录时间戳和用户信息                    │
└────────────────┬───────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│  Step 2: 知识图谱构建                   │
│  ├─ 用户节点                            │
│  ├─ 评论节点 (包含情感标签)              │
│  ├─ 话题/实体节点                       │
│  └─ 各类关系边                          │
└────────────────┬───────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│  Step 3: KGCN 时序图构建                │
│  ├─ 时刻 t-7 (7天前)                    │
│  ├─ 时刻 t-3 (3天前)                    │
│  ├─ 时刻 t-1 (昨天)                     │
│  └─ 时刻 t (今天)                       │
└────────────────┬───────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│  Step 4: KGCN 模型预测                 │
│  ├─ 预测用户明天的情感倾向               │
│  ├─ 预测评论热点话题的情感轨迹           │
│  └─ 识别异常/升级信号                   │
└────────────────┬───────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│  Step 5: 风险预警                       │
│  ├─ 情感急剧转负 → 黄色预警              │
│  ├─ 负面情感扩大 + 头部用户参与 → 橙色预警│
│  └─ 极端情感 + 快速传播 → 红色预警      │
└────────────────┬───────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│  Web 前端展示                            │
│  ├─ 预测曲线 (历史+预测)                 │
│  ├─ 风险等级指示                        │
│  └─ 实时预警通知                        │
└──────────────────────────────────────────┘
```

---

## 第二部分：知识图谱构建

### 2.1 舆情场景的知识图谱设计

#### 实体类型 (Node Types)

```python
class NodeType:
    USER = "user"              # 微博用户
    COMMENT = "comment"        # 评论/微博内容
    TOPIC = "topic"            # 话题标签
    ENTITY = "entity"          # 话题中的实体（如产品、公司、人物）
    KEYWORD = "keyword"        # 关键词（通过NER或TF-IDF提取）
    SENTIMENT = "sentiment"    # 情感标签 (positive/neutral/negative)
    TIME_SLOT = "time_slot"    # 时间槽（小时）
```

#### 关系类型 (Edge Types)

```python
class RelationType:
    # 用户行为
    USER_WRITES = "writes"              # 用户-编写-评论
    USER_LIKES = "likes"                # 用户-点赞-评论
    USER_REPLIES = "replies"            # 用户-回复-评论 (时序)
    USER_INTERACTS_WITH = "interacts"   # 用户-互动-用户
    
    # 评论属性
    COMMENT_HAS_SENTIMENT = "has_sentiment"  # 评论-具有-情感
    COMMENT_MENTIONS_TOPIC = "mentions"      # 评论-提及-话题
    COMMENT_MENTIONS_ENTITY = "mentions_entity"  # 评论-提及-实体
    COMMENT_CONTAINS_KEYWORD = "contains_keyword"  # 评论-包含-关键词
    COMMENT_AT_TIME = "at_time"          # 评论-发表于-时间
    
    # 时序关系（关键！）
    COMMENT_FOLLOWS = "follows"          # 评论1-时间后-评论2
    USER_SENTIMENT_CHANGES = "sentiment_changes"  # 用户的情感转变
    TOPIC_EMOTION_EVOLVES = "emotion_evolves"    # 话题情感演变
```

### 2.2 知识图谱构建代码

```python
"""
kg_builder.py - 为舆情数据构建知识图谱
"""

import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import jieba
from dataclasses import dataclass

@dataclass
class Comment:
    """评论数据模型"""
    comment_id: str
    user_id: str
    content: str
    content_clean: str
    sentiment_score: float  # BERT输出：-1(负) ~ 0(中立) ~ 1(正)
    sentiment_label: str    # "negative" / "neutral" / "positive"
    created_at: datetime
    likes: int
    replies: int
    keywords: List[str]     # 从content_clean提取的关键词
    entities: List[str]     # 从content中提取的实体
    topic_id: str

@dataclass  
class User:
    """用户数据模型"""
    user_id: str
    username: str
    followers: int
    is_certified: bool  # 认证用户
    comments: List[Comment]  # 该用户的所有评论

class SentimentKnowledgeGraph:
    """舆情知识图谱构建器"""
    
    def __init__(self):
        """初始化有向图"""
        self.graph = nx.DiGraph()
        # 节点属性记录
        self.node_attrs = {}
        # 边属性记录
        self.edge_attrs = {}
        
    def add_comment_node(self, comment: Comment):
        """添加评论节点"""
        node_id = f"comment_{comment.comment_id}"
        
        self.graph.add_node(node_id, node_type="comment")
        self.node_attrs[node_id] = {
            "comment_id": comment.comment_id,
            "content": comment.content_clean,
            "sentiment_score": comment.sentiment_score,
            "sentiment_label": comment.sentiment_label,
            "created_at": comment.created_at,
            "user_id": comment.user_id,
            "topic_id": comment.topic_id,
            "likes": comment.likes,
            "replies": comment.replies,
        }
        
    def add_user_node(self, user: User):
        """添加用户节点"""
        node_id = f"user_{user.user_id}"
        
        self.graph.add_node(node_id, node_type="user")
        self.node_attrs[node_id] = {
            "user_id": user.user_id,
            "username": user.username,
            "followers": user.followers,
            "is_certified": user.is_certified,
        }
        
    def add_topic_node(self, topic_id: str, topic_name: str):
        """添加话题节点"""
        node_id = f"topic_{topic_id}"
        
        self.graph.add_node(node_id, node_type="topic")
        self.node_attrs[node_id] = {
            "topic_id": topic_id,
            "topic_name": topic_name,
        }
        
    def add_entity_node(self, entity_name: str, entity_type: str):
        """添加实体节点（如产品、公司名称）"""
        node_id = f"entity_{entity_name}"
        
        if node_id not in self.graph:
            self.graph.add_node(node_id, node_type="entity")
            self.node_attrs[node_id] = {
                "entity_name": entity_name,
                "entity_type": entity_type,  # "product" / "company" / "person"
            }
            
    def add_keyword_node(self, keyword: str):
        """添加关键词节点"""
        node_id = f"keyword_{keyword}"
        
        if node_id not in self.graph:
            self.graph.add_node(node_id, node_type="keyword")
            self.node_attrs[node_id] = {"keyword": keyword}
            
    def add_sentiment_node(self, sentiment_label: str):
        """添加情感节点"""
        node_id = f"sentiment_{sentiment_label}"
        
        if node_id not in self.graph:
            self.graph.add_node(node_id, node_type="sentiment")
            self.node_attrs[node_id] = {"sentiment": sentiment_label}
    
    def connect_user_writes_comment(self, user_id: str, comment: Comment, weight=1.0):
        """用户-编写-评论"""
        user_node = f"user_{user_id}"
        comment_node = f"comment_{comment.comment_id}"
        
        self.graph.add_edge(user_node, comment_node, relation_type="writes", weight=weight)
        
    def connect_comment_has_sentiment(self, comment: Comment):
        """评论-具有-情感"""
        comment_node = f"comment_{comment.comment_id}"
        sentiment_node = f"sentiment_{comment.sentiment_label}"
        
        # 权重：情感分数的绝对值（越强的情感，权重越大）
        weight = abs(comment.sentiment_score)
        
        self.graph.add_edge(
            comment_node, 
            sentiment_node, 
            relation_type="has_sentiment",
            weight=weight,
            score=comment.sentiment_score
        )
        
    def connect_comment_mentions_entity(self, comment_id: str, entity_name: str):
        """评论-提及-实体"""
        comment_node = f"comment_{comment_id}"
        entity_node = f"entity_{entity_name}"
        
        self.graph.add_edge(
            comment_node,
            entity_node,
            relation_type="mentions_entity",
            weight=1.0
        )
        
    def connect_comment_contains_keyword(self, comment_id: str, keyword: str, freq=1):
        """评论-包含-关键词"""
        comment_node = f"comment_{comment_id}"
        keyword_node = f"keyword_{keyword}"
        
        self.graph.add_edge(
            comment_node,
            keyword_node,
            relation_type="contains_keyword",
            weight=freq  # 关键词出现频次作为权重
        )
        
    def connect_comment_mentions_topic(self, comment_id: str, topic_id: str):
        """评论-提及-话题"""
        comment_node = f"comment_{comment_id}"
        topic_node = f"topic_{topic_id}"
        
        self.graph.add_edge(
            comment_node,
            topic_node,
            relation_type="mentions_topic",
            weight=1.0
        )
        
    def connect_comments_temporal(self, comment1: Comment, comment2: Comment):
        """
        建立评论间的时序关系
        如果 comment1 时间早于 comment2，则添加有向边 comment1 -> comment2
        """
        if comment1.created_at < comment2.created_at:
            comment1_node = f"comment_{comment1.comment_id}"
            comment2_node = f"comment_{comment2.comment_id}"
            
            time_delta = (comment2.created_at - comment1.created_at).total_seconds()
            
            self.graph.add_edge(
                comment1_node,
                comment2_node,
                relation_type="follows",
                weight=1.0 / (1 + time_delta / 3600),  # 时间越接近，权重越大
                time_delta_hours=time_delta / 3600
            )
    
    def connect_user_sentiment_trajectory(self, user_id: str, comments: List[Comment]):
        """
        建立用户的情感轨迹（关键！）
        同一用户的评论按时间顺序连接
        """
        sorted_comments = sorted(comments, key=lambda c: c.created_at)
        
        for i in range(len(sorted_comments) - 1):
            comment1 = sorted_comments[i]
            comment2 = sorted_comments[i + 1]
            
            self.connect_comments_temporal(comment1, comment2)
            
            # 额外：记录情感转变
            sentiment_change = comment2.sentiment_score - comment1.sentiment_score
            
            # 如果情感有明显变化（±0.2），标记为"sentiment_changes"关系
            if abs(sentiment_change) > 0.2:
                comment1_node = f"comment_{comment1.comment_id}"
                comment2_node = f"comment_{comment2.comment_id}"
                
                edge_data = self.graph[comment1_node][comment2_node]
                edge_data['sentiment_change'] = sentiment_change
                edge_data['is_escalation'] = sentiment_change < -0.2  # 情感转负
    
    def connect_user_interactions(self, user1_id: str, user2_id: str, interaction_type="reply"):
        """两个用户之间的互动关系"""
        user1_node = f"user_{user1_id}"
        user2_node = f"user_{user2_id}"
        
        if user1_node in self.graph and user2_node in self.graph:
            self.graph.add_edge(
                user1_node,
                user2_node,
                relation_type="interacts",
                interaction_type=interaction_type,
                weight=1.0
            )
    
    def build_from_comments(self, comments: List[Comment], users: Dict[str, User], 
                           topic_id: str, topic_name: str):
        """
        从评论列表构建完整的知识图谱
        
        Args:
            comments: 评论列表
            users: 用户字典 {user_id -> User}
            topic_id: 话题ID
            topic_name: 话题名称
        """
        # Step 1: 添加话题节点
        self.add_topic_node(topic_id, topic_name)
        
        # Step 2: 添加用户和评论节点
        for comment in comments:
            self.add_comment_node(comment)
            
            if comment.user_id in users:
                user = users[comment.user_id]
                self.add_user_node(user)
                
                # 连接用户-评论
                self.connect_user_writes_comment(comment.user_id, comment)
        
        # Step 3: 添加评论属性节点并连接
        for comment in comments:
            # 情感节点
            self.add_sentiment_node(comment.sentiment_label)
            self.connect_comment_has_sentiment(comment)
            
            # 话题连接
            self.connect_comment_mentions_topic(comment.comment_id, topic_id)
            
            # 关键词节点（使用jieba分词）
            if comment.content_clean:
                words = jieba.cut(comment.content_clean)
                for word in words:
                    if len(word) > 1:  # 过滤单字
                        self.add_keyword_node(word)
                        self.connect_comment_contains_keyword(comment.comment_id, word)
            
            # 实体节点（这里用关键词做代替，实际应该用NER提取）
            if comment.entities:
                for entity in comment.entities:
                    self.add_entity_node(entity, "entity")
                    self.connect_comment_mentions_entity(comment.comment_id, entity)
        
        # Step 4: 建立时序关系（最重要！）
        # 全局时序：所有评论按发表时间顺序连接
        sorted_comments = sorted(comments, key=lambda c: c.created_at)
        for i in range(len(sorted_comments) - 1):
            self.connect_comments_temporal(sorted_comments[i], sorted_comments[i + 1])
        
        # 用户维度时序：同一用户的评论按时间连接
        for user_id, user in users.items():
            if user.comments:
                self.connect_user_sentiment_trajectory(user_id, user.comments)
        
        return self.graph
    
    def get_graph_stats(self):
        """获取图的统计信息"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
        }
    
    def _count_node_types(self):
        """统计各类节点的数量"""
        counts = {}
        for node, attr in self.graph.nodes(data=True):
            node_type = attr.get("node_type", "unknown")
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _count_edge_types(self):
        """统计各类边的数量"""
        counts = {}
        for u, v, attr in self.graph.edges(data=True):
            rel_type = attr.get("relation_type", "unknown")
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
```

### 2.3 使用示例

```python
"""
example_kg_build.py - 知识图谱构建示例
"""

from datetime import datetime, timedelta
import random

# 模拟数据
def create_mock_data():
    """创建模拟评论和用户数据"""
    
    comments = []
    users = {}
    
    # 创建3个用户
    for i in range(3):
        user_id = f"user_{i}"
        users[user_id] = User(
            user_id=user_id,
            username=f"用户{i}",
            followers=random.randint(100, 10000),
            is_certified=i == 0,  # 第一个是认证用户
            comments=[]
        )
    
    # 为每个用户创建多条评论（模拟时序数据）
    base_time = datetime.now() - timedelta(days=7)
    
    sentiment_scores = [
        -0.8, -0.6, -0.5,  # 负面情感
        -0.2,  # 轻微负面
        0.0, 0.1,  # 中立
        0.5, 0.8,  # 正面
    ]
    
    comment_id = 0
    for user_idx, (user_id, user) in enumerate(users.items()):
        for day in range(7):
            # 每个用户每天2条评论
            for hour in range(2):
                comment = Comment(
                    comment_id=f"comment_{comment_id}",
                    user_id=user_id,
                    content=f"用户{user_idx}在第{day}天第{hour}小时的评论",
                    content_clean=f"评论 {random.choice(['产品', '服务', '公司'])} "
                                 f"{random.choice(['很好', '很差', '一般'])}",
                    sentiment_score=random.choice(sentiment_scores),
                    sentiment_label="positive" if random.random() > 0.5 else "negative",
                    created_at=base_time + timedelta(days=day, hours=hour),
                    likes=random.randint(0, 1000),
                    replies=random.randint(0, 100),
                    keywords=["产品", "服务"],
                    entities=["某品牌", "某公司"],
                    topic_id="topic_001"
                )
                
                comments.append(comment)
                user.comments.append(comment)
                comment_id += 1
    
    return comments, users

# 构建知识图谱
if __name__ == "__main__":
    comments, users = create_mock_data()
    
    kg = SentimentKnowledgeGraph()
    graph = kg.build_from_comments(
        comments=comments,
        users=users,
        topic_id="topic_001",
        topic_name="某热点话题"
    )
    
    stats = kg.get_graph_stats()
    print(f"知识图谱统计:")
    print(f"  节点数: {stats['num_nodes']}")
    print(f"  边数: {stats['num_edges']}")
    print(f"  节点类型分布: {stats['node_types']}")
    print(f"  边类型分布: {stats['edge_types']}")
```

---

## 第三部分：KGCN 时序图快照

### 3.1 时序图快照的概念

关键点：**KGCN 本身不是时序模型**，但我们可以构建多个时间切片的图快照来模拟时序。

```
Day 1         Day 3         Day 5         Day 7 (今天)
┌────┐       ┌────┐       ┌────┐       ┌────────┐
│G_1 │       │G_3 │       │G_5 │       │G_7     │ ← 当前图
└────┘       └────┘       └────┘       └────────┘
  ↓            ↓             ↓              ↓
 用户1-评论     用户1-评论    用户1-评论    用户1-评论
 情感: -0.3   情感: -0.5   情感: -0.7   情感: -0.9
              ↑ 情感升级信号
                负面积累

KGCN预测逻辑：
输入：G_5, G_7 (最近两个快照的聚合信息)
输出：预测 Day 9 (t+2) 的用户情感走向 + 话题风险等级
```

### 3.2 时序快照构建代码

```python
"""
temporal_snapshots.py - 构建时序图快照
"""

from datetime import datetime, timedelta
from typing import List, Dict
import copy

class TemporalGraphSnapshot:
    """时序图快照"""
    
    def __init__(self, timestamp: datetime, graph):
        self.timestamp = timestamp
        self.graph = copy.deepcopy(graph)
        
        # 计算快照统计
        self.num_comments = len([n for n in self.graph.nodes() if n.startswith("comment_")])
        self.num_users = len([n for n in self.graph.nodes() if n.startswith("user_")])
        self.avg_sentiment = self._compute_avg_sentiment()
        self.sentiment_distribution = self._compute_sentiment_distribution()
        
    def _compute_avg_sentiment(self) -> float:
        """计算快照中所有评论的平均情感"""
        sentiment_scores = []
        for node, attr in self.graph.nodes(data=True):
            if node.startswith("comment_"):
                if 'sentiment_score' in attr:
                    sentiment_scores.append(attr['sentiment_score'])
        
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    def _compute_sentiment_distribution(self) -> Dict[str, float]:
        """计算快照中的情感分布"""
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        total = 0
        
        for node, attr in self.graph.nodes(data=True):
            if node.startswith("comment_"):
                label = attr.get('sentiment_label', 'neutral')
                distribution[label] += 1
                total += 1
        
        if total > 0:
            for key in distribution:
                distribution[key] /= total
        
        return distribution
    
    def get_high_risk_users(self, neg_threshold=0.6) -> List[str]:
        """
        获取高风险用户（多次发表极端负面评论）
        """
        user_neg_count = {}
        
        for node, attr in self.graph.nodes(data=True):
            if node.startswith("comment_"):
                if attr.get('sentiment_score', 0) < -neg_threshold:
                    user_id = attr.get('user_id')
                    if user_id:
                        user_neg_count[user_id] = user_neg_count.get(user_id, 0) + 1
        
        # 返回负面评论数≥2的用户
        return [uid for uid, count in user_neg_count.items() if count >= 2]
    
    def get_escalation_signals(self) -> List[Dict]:
        """
        检测情感升级信号（从快照的角度）
        返回：[{user_id, escalation_degree, comment_ids}, ...]
        """
        signals = []
        
        for user_id in set([attr.get('user_id') for _, attr in self.graph.nodes(data=True) 
                           if attr.get('node_type') == 'comment']):
            if not user_id:
                continue
            
            # 获取该用户在本快照中的所有评论
            user_comments = []
            for node, attr in self.graph.nodes(data=True):
                if node.startswith("comment_") and attr.get('user_id') == user_id:
                    user_comments.append({
                        'comment_id': attr.get('comment_id'),
                        'sentiment_score': attr.get('sentiment_score'),
                        'created_at': attr.get('created_at'),
                    })
            
            if len(user_comments) >= 2:
                # 按时间排序
                user_comments.sort(key=lambda c: c['created_at'])
                
                # 计算情感趋势
                sentiment_trend = [c['sentiment_score'] for c in user_comments]
                
                # 检测升级：最后3条评论的平均值 < 倒数4-6条的平均值
                if len(sentiment_trend) >= 6:
                    recent_avg = sum(sentiment_trend[-3:]) / 3
                    older_avg = sum(sentiment_trend[-6:-3]) / 3
                    escalation = older_avg - recent_avg
                    
                    if escalation > 0.3:  # 情感下降超过0.3
                        signals.append({
                            'user_id': user_id,
                            'escalation_degree': escalation,
                            'comment_ids': [c['comment_id'] for c in user_comments[-3:]],
                            'recent_avg_sentiment': recent_avg,
                        })
        
        # 按升级程度排序
        signals.sort(key=lambda s: s['escalation_degree'], reverse=True)
        return signals


class TemporalGraphSequence:
    """时序图快照序列"""
    
    def __init__(self, time_window_days=7, interval_days=1):
        """
        Args:
            time_window_days: 总时间窗口（天数）
            interval_days: 快照间隔（天数）
        """
        self.time_window_days = time_window_days
        self.interval_days = interval_days
        self.snapshots: List[TemporalGraphSnapshot] = []
        
    def add_snapshot(self, snapshot: TemporalGraphSnapshot):
        """添加图快照"""
        self.snapshots.append(snapshot)
        # 按时间排序
        self.snapshots.sort(key=lambda s: s.timestamp)
    
    def get_sentiment_evolution(self) -> Dict:
        """获取情感演变曲线"""
        evolution = {
            'timestamps': [],
            'avg_sentiments': [],
            'pos_ratios': [],
            'neu_ratios': [],
            'neg_ratios': [],
        }
        
        for snapshot in self.snapshots:
            evolution['timestamps'].append(snapshot.timestamp.isoformat())
            evolution['avg_sentiments'].append(snapshot.avg_sentiment)
            
            dist = snapshot.sentiment_distribution
            evolution['pos_ratios'].append(dist.get('positive', 0))
            evolution['neu_ratios'].append(dist.get('neutral', 0))
            evolution['neg_ratios'].append(dist.get('negative', 0))
        
        return evolution
    
    def get_escalation_trajectory(self) -> Dict:
        """获取升级轨迹"""
        trajectory = {
            'timestamps': [],
            'high_risk_user_counts': [],
            'avg_escalation_signals': [],
        }
        
        for snapshot in self.snapshots:
            trajectory['timestamps'].append(snapshot.timestamp.isoformat())
            
            high_risk_users = snapshot.get_high_risk_users()
            trajectory['high_risk_user_counts'].append(len(high_risk_users))
            
            signals = snapshot.get_escalation_signals()
            avg_escalation = sum([s['escalation_degree'] for s in signals]) / len(signals) \
                           if signals else 0
            trajectory['avg_escalation_signals'].append(avg_escalation)
        
        return trajectory
    
    def detect_anomalies(self) -> List[Dict]:
        """
        检测异常点（可能的二次发酵信号）
        """
        anomalies = []
        
        if len(self.snapshots) < 2:
            return anomalies
        
        for i in range(1, len(self.snapshots)):
            prev_snapshot = self.snapshots[i - 1]
            curr_snapshot = self.snapshots[i]
            
            # 异常指标1：负面情感突增
            neg_increase = (curr_snapshot.sentiment_distribution['negative'] - 
                          prev_snapshot.sentiment_distribution['negative'])
            
            if neg_increase > 0.15:  # 负面增加超过15%
                anomalies.append({
                    'type': 'negative_sentiment_spike',
                    'timestamp': curr_snapshot.timestamp,
                    'severity': min(neg_increase / 0.3, 1.0),  # 正则化到[0,1]
                    'description': f"负面情感快速增加 {neg_increase:.1%}"
                })
            
            # 异常指标2：高风险用户激增
            prev_high_risk = snapshot.get_high_risk_users()
            curr_high_risk = curr_snapshot.get_high_risk_users()
            
            risk_increase = len(curr_high_risk) - len(prev_high_risk)
            if risk_increase >= 2:
                anomalies.append({
                    'type': 'high_risk_user_surge',
                    'timestamp': curr_snapshot.timestamp,
                    'severity': min(risk_increase / 5, 1.0),
                    'new_users': list(set(curr_high_risk) - set(prev_high_risk)),
                    'description': f"高风险用户增加 {risk_increase} 人"
                })
            
            # 异常指标3：情感快速转负
            sentiment_drop = prev_snapshot.avg_sentiment - curr_snapshot.avg_sentiment
            if sentiment_drop > 0.3:
                anomalies.append({
                    'type': 'sentiment_plunge',
                    'timestamp': curr_snapshot.timestamp,
                    'severity': min(sentiment_drop / 0.6, 1.0),
                    'description': f"情感快速下降 {sentiment_drop:.2f}"
                })
        
        return anomalies
```

### 3.3 使用示例

```python
"""
example_temporal_snapshots.py - 时序快照使用示例
"""

def build_daily_snapshots(comments: List[Comment], days=7) -> TemporalGraphSequence:
    """
    从评论列表构建每日图快照
    """
    sequence = TemporalGraphSequence(time_window_days=days, interval_days=1)
    
    # 按日期分组评论
    from datetime import date
    comments_by_day = {}
    
    for comment in comments:
        day_key = comment.created_at.date()
        if day_key not in comments_by_day:
            comments_by_day[day_key] = []
        comments_by_day[day_key].append(comment)
    
    # 为每一天构建图快照
    for day, day_comments in sorted(comments_by_day.items()):
        # 构建该天的知识图谱
        kg = SentimentKnowledgeGraph()
        # （这里省略详细的构建过程）
        graph = kg.build_from_comments(
            comments=day_comments,
            users={...},  # 对应的用户
            topic_id="topic_001",
            topic_name="某热点话题"
        )
        
        # 创建快照
        snapshot = TemporalGraphSnapshot(
            timestamp=datetime.combine(day, datetime.min.time()),
            graph=graph
        )
        
        sequence.add_snapshot(snapshot)
    
    return sequence


# 使用
if __name__ == "__main__":
    # 假设已有7天的评论数据
    sequence = build_daily_snapshots(all_comments, days=7)
    
    # 获取情感演变
    evolution = sequence.get_sentiment_evolution()
    print("情感演变:")
    for ts, sentiment in zip(evolution['timestamps'], evolution['avg_sentiments']):
        print(f"  {ts}: {sentiment:.3f}")
    
    # 检测异常（二次发酵信号）
    anomalies = sequence.detect_anomalies()
    print(f"\n检测到 {len(anomalies)} 个异常信号:")
    for anomaly in anomalies:
        print(f"  [{anomaly['type']}] {anomaly['description']}")
```

---

## 第四部分：KGCN 模型集成

### 4.1 KGCN 模型架构回顾

```python
"""
基于KGCN论文的实现，针对舆情优化
"""

class KGCNLayer(nn.Module):
    """KGCN 单层"""
    
    def __init__(self, embedding_dim, hidden_dim, aggregator_type="sum"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 线性变换矩阵
        if aggregator_type == "sum":
            self.W = nn.Linear(embedding_dim, hidden_dim)
            self.b = nn.Parameter(torch.zeros(hidden_dim))
        elif aggregator_type == "concat":
            self.W = nn.Linear(2 * embedding_dim, hidden_dim)
            self.b = nn.Parameter(torch.zeros(hidden_dim))
        
        self.aggregator_type = aggregator_type
        self.activation = nn.ReLU()
        
    def forward(self, entity_embedding, neighbor_embeddings, attention_weights):
        """
        Args:
            entity_embedding: [embedding_dim] - 实体自身的嵌入
            neighbor_embeddings: [num_neighbors, embedding_dim] - 邻域嵌入
            attention_weights: [num_neighbors] - 邻域权重（基于用户和关系）
        
        Returns:
            updated_embedding: [hidden_dim] - 更新后的表示
        """
        # 加权聚合邻域
        weighted_neighbors = (attention_weights.unsqueeze(1) * neighbor_embeddings).sum(dim=0)
        # [embedding_dim]
        
        if self.aggregator_type == "sum":
            combined = entity_embedding + weighted_neighbors
            output = self.W(combined) + self.b
        elif self.aggregator_type == "concat":
            combined = torch.cat([entity_embedding, weighted_neighbors], dim=0)
            output = self.W(combined) + self.b
        
        return self.activation(output)


class SentimentKGCN(nn.Module):
    """针对舆情情感预测的 KGCN 模型"""
    
    def __init__(self, config):
        super().__init__()
        
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        
        # 嵌入层
        self.entity_embedding = nn.Embedding(config['num_entities'], self.embedding_dim)
        self.relation_embedding = nn.Embedding(config['num_relations'], self.embedding_dim)
        self.user_embedding = nn.Embedding(config['num_users'], self.embedding_dim)
        
        # KGCN 层
        self.kgcn_layers = nn.ModuleList([
            KGCNLayer(self.embedding_dim, self.hidden_dim, aggregator_type="sum")
            for _ in range(self.num_layers)
        ])
        
        # 用户-关系注意力（计算权重）
        self.relation_attention = nn.Sequential(
            nn.Linear(self.embedding_dim + self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # 3个情感类别：负/中/正
        )
        
    def compute_relation_attention(self, user_embedding, relation_embedding):
        """
        计算用户对关系的关注度
        π_r^u = attention(user, relation)
        
        Returns: [1] - 标量注意力分数
        """
        combined = torch.cat([user_embedding, relation_embedding], dim=-1)
        attention_score = self.relation_attention(combined)
        return torch.sigmoid(attention_score)
    
    def forward(self, user_id, entity_ids, relation_ids, neighbor_ids, graph):
        """
        前向传播
        
        Args:
            user_id: 用户ID
            entity_ids: 主要实体ID列表 [num_entities]
            relation_ids: 关系ID列表 [num_relations]
            neighbor_ids: 邻域实体ID列表 [num_entities, num_neighbors]
            graph: 知识图谱
        
        Returns:
            sentiment_logits: [num_entities, 3] - 每个实体的情感预测
        """
        # 获取嵌入
        user_emb = self.user_embedding(torch.tensor([user_id]))  # [1, embedding_dim]
        entity_embs = self.entity_embedding(entity_ids)  # [num_entities, embedding_dim]
        relation_embs = self.relation_embedding(relation_ids)  # [num_relations, embedding_dim]
        
        # KGCN 聚合
        updated_embs = entity_embs
        
        for layer_idx, kgcn_layer in enumerate(self.kgcn_layers):
            layer_embs = []
            
            for entity_idx in range(len(entity_ids)):
                entity_id = entity_ids[entity_idx]
                entity_emb = updated_embs[entity_idx]
                
                # 获取邻域
                neighbors = neighbor_ids[entity_idx]
                neighbor_embs = self.entity_embedding(neighbors)  # [num_neighbors, embedding_dim]
                
                # 计算注意力权重（简化：使用关系作为加权信息）
                if len(neighbors) > 0:
                    # 对邻域进行softmax加权
                    relation_scores = []
                    for rel_emb in relation_embs:
                        score = self.compute_relation_attention(user_emb.squeeze(0), rel_emb)
                        relation_scores.append(score)
                    
                    attention_weights = torch.softmax(torch.tensor(relation_scores), dim=0)
                    
                    # KGCN 聚合
                    updated_emb = kgcn_layer(entity_emb, neighbor_embs, attention_weights)
                else:
                    updated_emb = entity_emb
                
                layer_embs.append(updated_emb)
            
            updated_embs = torch.stack(layer_embs)
        
        # 预测头
        sentiment_logits = self.prediction_head(updated_embs)  # [num_entities, 3]
        
        return sentiment_logits
```

### 4.2 KGCN 的时序应用策略

**关键洞察**：KGCN 用于**聚合多个快照的信息**来预测未来情感

```python
"""
temporal_kgcn_prediction.py - 用时序快照和KGCN进行预测
"""

class TemporalKGCNPredictor:
    """
    利用时序图快照序列进行KGCN情感预测
    
    策略：
    1. 对每个快照执行KGCN（得到该时刻的实体表示）
    2. 在LSTM中串联多个快照的表示（捕捉时间变化）
    3. 输出未来t+1时刻的情感预测
    """
    
    def __init__(self, kgcn_model, lstm_hidden_dim=128):
        self.kgcn_model = kgcn_model
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # LSTM 用于聚合时序KGCN输出
        self.lstm = nn.LSTM(
            input_size=kgcn_model.hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 预测头：从LSTM隐态预测未来情感
        self.forecast_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3个情感类别
        )
    
    def forward(self, snapshot_sequence: TemporalGraphSequence, 
                user_id, topic_entities, forecast_days=3):
        """
        基于时序快照序列进行情感预测
        
        Args:
            snapshot_sequence: 时序图快照序列
            user_id: 用户ID
            topic_entities: 话题相关实体列表
            forecast_days: 预测天数
        
        Returns:
            predictions: {
                'history_sentiments': [...],  # 历史情感
                'forecast_sentiments': [...],  # 预测情感
                'confidence_interval': [...],  # 置信区间
                'anomaly_alert': bool,  # 是否有异常信号
            }
        """
        predictions = {
            'history_sentiments': [],
            'forecast_sentiments': [],
            'confidence_interval': [],
            'risk_signals': [],
        }
        
        # Step 1: 对每个快照执行KGCN，得到该时刻的实体表示
        snapshot_representations = []
        
        for snapshot in snapshot_sequence.snapshots:
            # 从快照的图中提取信息
            entity_ids = []  # 从图中获取实体ID
            relation_ids = []  # 从图中获取关系ID
            neighbor_ids = []  # 邻域ID
            
            # （这里省略具体的图遍历逻辑）
            
            # 执行KGCN
            with torch.no_grad():
                kgcn_output = self.kgcn_model(
                    user_id=user_id,
                    entity_ids=torch.tensor(entity_ids),
                    relation_ids=torch.tensor(relation_ids),
                    neighbor_ids=torch.tensor(neighbor_ids),
                    graph=snapshot.graph
                )
            
            # kgcn_output: [num_entities, 3]
            # 取平均作为该快照的整体表示
            snapshot_repr = kgcn_output.mean(dim=0).unsqueeze(0)  # [1, 3]
            snapshot_representations.append(snapshot_repr)
            
            # 记录历史情感
            history_sentiment = torch.softmax(kgcn_output.mean(dim=0), dim=-1)
            predictions['history_sentiments'].append({
                'timestamp': snapshot.timestamp,
                'pos_prob': history_sentiment[2].item(),
                'neu_prob': history_sentiment[1].item(),
                'neg_prob': history_sentiment[0].item(),
            })
        
        # Step 2: 通过LSTM聚合时序信息
        snapshot_seq = torch.cat(snapshot_representations, dim=0).unsqueeze(0)  # [1, T, 3]
        
        lstm_out, (h_n, c_n) = self.lstm(snapshot_seq)
        # lstm_out: [1, T, lstm_hidden_dim]
        # h_n: [2, 1, lstm_hidden_dim] (最后一层的隐态)
        
        final_hidden = h_n[-1].squeeze(0)  # [lstm_hidden_dim]
        
        # Step 3: 预测未来情感
        for day in range(forecast_days):
            # 简单策略：使用最终隐态预测未来每一天
            forecast_logits = self.forecast_head(final_hidden)
            forecast_prob = torch.softmax(forecast_logits, dim=-1)
            
            predictions['forecast_sentiments'].append({
                'day': day + 1,
                'pos_prob': forecast_prob[2].item(),
                'neu_prob': forecast_prob[1].item(),
                'neg_prob': forecast_prob[0].item(),
            })
            
            # 简单置信区间：基于负面概率
            neg_prob = forecast_prob[0].item()
            confidence = 0.95 - abs(neg_prob - 0.5) * 0.5  # 越确定置信度越低
            
            predictions['confidence_interval'].append({
                'day': day + 1,
                'upper': neg_prob + (1 - confidence) * 0.1,
                'lower': max(0, neg_prob - (1 - confidence) * 0.1),
            })
        
        # Step 4: 检测异常信号
        anomalies = snapshot_sequence.detect_anomalies()
        predictions['risk_signals'] = anomalies
        
        return predictions


def predict_sentiment_trend(comments: List[Comment], 
                           users: Dict[str, User],
                           topic_id: str,
                           topic_name: str,
                           forecast_days=3) -> Dict:
    """
    完整的预测管道：从评论到情感预测
    """
    
    # Step 1: 构建时序快照序列
    sequence = build_daily_snapshots(comments, days=7)
    
    # Step 2: 初始化KGCN模型
    config = {
        'embedding_dim': 32,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_entities': 500,  # 实体总数
        'num_relations': 10,  # 关系总数
        'num_users': len(users),  # 用户总数
    }
    
    kgcn_model = SentimentKGCN(config)
    kgcn_model.load_state_dict(torch.load('kgcn_model.pth'))  # 加载预训练权重
    
    # Step 3: 创建预测器
    predictor = TemporalKGCNPredictor(kgcn_model, lstm_hidden_dim=128)
    
    # Step 4: 执行预测
    # 对热点话题的关键用户进行预测
    top_users = get_top_active_users(users, top_k=5)
    
    all_predictions = {}
    for user in top_users:
        user_predictions = predictor.forward(
            snapshot_sequence=sequence,
            user_id=user.user_id,
            topic_entities=extract_topic_entities(comments),
            forecast_days=forecast_days
        )
        all_predictions[user.user_id] = user_predictions
    
    return {
        'topic_id': topic_id,
        'topic_name': topic_name,
        'forecast_days': forecast_days,
        'user_predictions': all_predictions,
        'overall_anomalies': sequence.detect_anomalies(),
    }
```

---

## 第五部分：集成到现有系统

### 5.1 数据库表设计

```sql
-- 添加到现有数据库

-- 预测结果表
CREATE TABLE tb_prediction (
    prediction_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    topic_id VARCHAR(100),
    timestamp DATETIME,
    
    -- 历史数据
    history_pos_ratio DECIMAL(5,4),
    history_neu_ratio DECIMAL(5,4),
    history_neg_ratio DECIMAL(5,4),
    
    -- 预测数据（未来3天）
    day1_pred_pos DECIMAL(5,4),
    day1_pred_neu DECIMAL(5,4),
    day1_pred_neg DECIMAL(5,4),
    day1_confidence DECIMAL(5,4),
    
    day2_pred_pos DECIMAL(5,4),
    day2_pred_neu DECIMAL(5,4),
    day2_pred_neg DECIMAL(5,4),
    day2_confidence DECIMAL(5,4),
    
    day3_pred_pos DECIMAL(5,4),
    day3_pred_neu DECIMAL(5,4),
    day3_pred_neg DECIMAL(5,4),
    day3_confidence DECIMAL(5,4),
    
    -- 风险指标
    escalation_score DECIMAL(5,4),  -- 升级倾向 (0-1)
    anomaly_detected BOOLEAN,
    anomaly_type VARCHAR(50),
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_topic_time (topic_id, timestamp),
    FOREIGN KEY (topic_id) REFERENCES tb_topic(topic_id)
);

-- 用户情感轨迹表
CREATE TABLE tb_user_sentiment_trajectory (
    trajectory_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(100),
    topic_id VARCHAR(100),
    
    -- 情感趋势指标
    avg_sentiment_score DECIMAL(5,4),  -- 平均情感分数
    sentiment_volatility DECIMAL(5,4),  -- 波动程度
    escalation_trend DECIMAL(5,4),  -- 升级趋势
    
    -- 预测
    next_day_sentiment_pred DECIMAL(5,4),
    next_day_risk_level INT,  # 1-5级
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_user_topic (user_id, topic_id),
    FOREIGN KEY (user_id) REFERENCES tb_user(user_id),
    FOREIGN KEY (topic_id) REFERENCES tb_topic(topic_id)
);

-- 异常信号表
CREATE TABLE tb_anomaly_signals (
    signal_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    topic_id VARCHAR(100),
    timestamp DATETIME,
    
    signal_type VARCHAR(50),  # 'negative_spike', 'escalation', 'user_surge'等
    severity DECIMAL(5,4),  # 严重程度 0-1
    description TEXT,
    
    affected_users INT,  # 涉及用户数
    affected_comments INT,  # 涉及评论数
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_topic_time (topic_id, timestamp),
    FOREIGN KEY (topic_id) REFERENCES tb_topic(topic_id)
);
```

### 5.2 FastAPI 接口集成

```python
"""
app/routes/prediction.py - 预测相关API接口
"""

from fastapi import APIRouter, Query
from app.services.kgcn_service import predict_sentiment_trend
from app.db import get_db
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/topics", tags=["prediction"])

@router.get("/{topic_id}/prediction")
async def get_sentiment_prediction(
    topic_id: str,
    days: int = Query(3, ge=1, le=7),
    db = Depends(get_db)
):
    """
    获取话题的情感预测结果
    
    Args:
        topic_id: 话题ID
        days: 预测天数 (1/3/7)
    
    Returns:
        {
            "code": 200,
            "data": {
                "topic_id": "...",
                "history": [...],  # 过去7天的历史数据
                "prediction": [...],  # 未来days天的预测
                "conclusion": {
                    "next_level": 3,
                    "escalation_risk": 0.75,
                    "risk_desc": "..."
                },
                "anomalies": [...]  # 检测到的异常信号
            }
        }
    """
    
    # Step 1: 从数据库获取评论数据（过去7天）
    seven_days_ago = datetime.now() - timedelta(days=7)
    comments = db.query(Comment).filter(
        Comment.topic_id == topic_id,
        Comment.created_at >= seven_days_ago
    ).all()
    
    if not comments:
        return {"code": 404, "message": "No comments found"}
    
    # Step 2: 获取用户信息
    user_ids = set([c.user_id for c in comments])
    users = {u.user_id: u for u in db.query(User).filter(
        User.user_id.in_(user_ids)
    ).all()}
    
    # Step 3: 获取话题信息
    topic = db.query(Topic).filter(Topic.topic_id == topic_id).first()
    
    # Step 4: 执行KGCN预测
    prediction_result = predict_sentiment_trend(
        comments=comments,
        users=users,
        topic_id=topic_id,
        topic_name=topic.name,
        forecast_days=days
    )
    
    # Step 5: 保存预测结果到数据库
    for user_id, user_preds in prediction_result['user_predictions'].items():
        # 构建数据库记录
        db_pred = Prediction(
            topic_id=topic_id,
            timestamp=datetime.now(),
            # ... 填充其他字段
        )
        db.add(db_pred)
    
    db.commit()
    
    # Step 6: 格式化响应
    response_data = {
        "topic_id": topic_id,
        "topic_name": topic.name,
        
        # 历史数据（最近7天）
        "history": [
            {
                "date": comment_day.isoformat(),
                "pos_ratio": ...,
                "neu_ratio": ...,
                "neg_ratio": ...,
                "is_predict": False
            }
            for comment_day in get_daily_sentiment(comments)
        ],
        
        # 预测数据
        "prediction": [
            {
                "date": (datetime.now() + timedelta(days=i+1)).isoformat(),
                "pos_ratio": ...,
                "neu_ratio": ...,
                "neg_ratio": ...,
                "confidence_upper": ...,
                "confidence_lower": ...,
                "is_predict": True
            }
            for i in range(days)
        ],
        
        # 风险结论
        "conclusion": {
            "next_level": calculate_next_level(prediction_result),
            "next_neg_ratio": ...,
            "escalation_risk": calculate_escalation_risk(prediction_result),
            "risk_desc": generate_risk_description(prediction_result)
        },
        
        # 异常信号
        "anomalies": prediction_result['overall_anomalies']
    }
    
    return {"code": 200, "data": response_data}


@router.get("/{topic_id}/anomalies")
async def get_anomaly_signals(
    topic_id: str,
    hours: int = Query(24, ge=1, le=168),
    db = Depends(get_db)
):
    """
    获取话题的异常信号
    """
    
    since = datetime.now() - timedelta(hours=hours)
    
    anomalies = db.query(AnomalySignal).filter(
        AnomalySignal.topic_id == topic_id,
        AnomalySignal.created_at >= since
    ).all()
    
    return {
        "code": 200,
        "data": {
            "topic_id": topic_id,
            "time_range": f"last_{hours}_hours",
            "anomaly_count": len(anomalies),
            "anomalies": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "type": a.signal_type,
                    "severity": float(a.severity),
                    "description": a.description,
                    "affected_users": a.affected_users,
                    "affected_comments": a.affected_comments
                }
                for a in anomalies
            ]
        }
    }


@router.get("/{topic_id}/user-trajectories")
async def get_user_sentiment_trajectories(
    topic_id: str,
    top_k: int = Query(10, ge=1, le=50),
    db = Depends(get_db)
):
    """
    获取话题中用户的情感轨迹（用于前端展示）
    """
    
    trajectories = db.query(UserSentimentTrajectory).filter(
        UserSentimentTrajectory.topic_id == topic_id
    ).order_by(
        UserSentimentTrajectory.escalation_trend.desc()
    ).limit(top_k).all()
    
    return {
        "code": 200,
        "data": {
            "topic_id": topic_id,
            "top_k": top_k,
            "trajectories": [
                {
                    "user_id": t.user_id,
                    "avg_sentiment": float(t.avg_sentiment_score),
                    "volatility": float(t.sentiment_volatility),
                    "escalation_trend": float(t.escalation_trend),
                    "next_day_pred": float(t.next_day_sentiment_pred),
                    "risk_level": t.next_day_risk_level
                }
                for t in trajectories
            ]
        }
    }
```

### 5.3 定时任务集成

```python
"""
app/tasks/prediction_tasks.py - APScheduler定时预测任务
"""

from apscheduler.schedulers.background import BackgroundScheduler
from app.services.kgcn_service import predict_sentiment_trend
from app.db import SessionLocal

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('cron', hour=2, minute=0)  # 每天凌晨2点
def daily_prediction_task():
    """
    每日定时预测任务（对标现有的 GNN预测任务）
    """
    
    db = SessionLocal()
    
    try:
        # 获取所有活跃话题
        active_topics = db.query(Topic).filter(
            Topic.updated_at >= datetime.now() - timedelta(hours=24)
        ).all()
        
        for topic in active_topics:
            # 获取该话题的近7天评论
            seven_days_ago = datetime.now() - timedelta(days=7)
            comments = db.query(Comment).filter(
                Comment.topic_id == topic.topic_id,
                Comment.created_at >= seven_days_ago
            ).all()
            
            if len(comments) < 10:  # 评论数太少，跳过
                continue
            
            # 获取用户信息
            user_ids = set([c.user_id for c in comments])
            users = {u.user_id: u for u in db.query(User).filter(
                User.user_id.in_(user_ids)
            ).all()}
            
            # 执行预测
            print(f"[KGCN预测] 话题 {topic.name} (ID:{topic.topic_id})")
            prediction_result = predict_sentiment_trend(
                comments=comments,
                users=users,
                topic_id=topic.topic_id,
                topic_name=topic.name,
                forecast_days=3
            )
            
            # 保存预测结果
            save_prediction_to_db(db, topic.topic_id, prediction_result)
            
            # 保存异常信号
            for anomaly in prediction_result['overall_anomalies']:
                save_anomaly_to_db(db, topic.topic_id, anomaly)
            
            # 保存用户轨迹
            for user_id, user_preds in prediction_result['user_predictions'].items():
                save_user_trajectory_to_db(db, user_id, topic.topic_id, user_preds)
            
            db.commit()
        
        print(f"[完成] 处理了 {len(active_topics)} 个话题的KGCN预测")
        
    except Exception as e:
        print(f"[错误] KGCN预测任务失败: {e}")
        db.rollback()
    finally:
        db.close()


def save_prediction_to_db(db, topic_id, prediction_result):
    """保存预测结果"""
    # ... 实现数据库保存逻辑


def save_anomaly_to_db(db, topic_id, anomaly):
    """保存异常信号"""
    # ... 实现数据库保存逻辑


def save_user_trajectory_to_db(db, user_id, topic_id, user_preds):
    """保存用户轨迹"""
    # ... 实现数据库保存逻辑


# 启动调度器
def start_scheduler():
    if not scheduler.running:
        scheduler.start()
        print("[启动] 后台任务调度器")
```

---

## 第六部分：前端展示优化

### 6.1 前端新增组件

```vue
<!-- SentimentTrendPrediction.vue - 情感趋势预测组件 -->
<template>
  <div class="prediction-container">
    <!-- 历史+预测折线图 -->
    <div class="chart-container">
      <h3>情感趋势预测（过去7天 + 未来3天）</h3>
      <div id="predictionChart"></div>
    </div>
    
    <!-- 预警信号展示 -->
    <div class="alerts-container" v-if="anomalies.length > 0">
      <h3>⚠️ 检测到的异常信号</h3>
      <div class="alert-cards">
        <div 
          v-for="alert in anomalies" 
          :key="alert.signal_id"
          :class="`alert-card severity-${Math.ceil(alert.severity * 5)}`"
        >
          <div class="alert-type">{{ formatAlertType(alert.signal_type) }}</div>
          <div class="alert-desc">{{ alert.description }}</div>
          <div class="alert-meta">
            <span>影响评论: {{ alert.affected_comments }}</span>
            <span>影响用户: {{ alert.affected_users }}</span>
            <span>严重程度: {{ (alert.severity * 100).toFixed(0) }}%</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 用户情感轨迹 -->
    <div class="user-trajectories-container">
      <h3>高风险用户情感轨迹</h3>
      <table class="user-table">
        <thead>
          <tr>
            <th>用户</th>
            <th>平均情感</th>
            <th>波动程度</th>
            <th>升级趋势</th>
            <th>明日预测</th>
            <th>风险等级</th>
          </tr>
        </thead>
        <tbody>
          <tr 
            v-for="traj in userTrajectories" 
            :key="traj.user_id"
            :class="`risk-level-${traj.risk_level}`"
          >
            <td>{{ traj.user_id }}</td>
            <td>{{ (traj.avg_sentiment * 100).toFixed(1) }}%</td>
            <td>{{ (traj.volatility * 100).toFixed(1) }}%</td>
            <td>
              <span :class="traj.escalation_trend > 0 ? 'up' : 'down'">
                {{ (traj.escalation_trend * 100).toFixed(1) }}%
              </span>
            </td>
            <td>{{ (traj.next_day_pred * 100).toFixed(1) }}%</td>
            <td>
              <span :class="`level-badge level-${traj.risk_level}`">
                L{{ traj.risk_level }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    
    <!-- 预测结论 -->
    <div class="conclusion-container">
      <h3>预测结论</h3>
      <div class="conclusion-card">
        <div class="metric">
          <span class="label">明日舆情等级:</span>
          <span :class="`level-badge level-${conclusion.next_level}`">
            L{{ conclusion.next_level }}
          </span>
        </div>
        <div class="metric">
          <span class="label">消极情感预测:</span>
          <span>{{ (conclusion.next_neg_ratio * 100).toFixed(1) }}%</span>
        </div>
        <div class="metric">
          <span class="label">升级风险:</span>
          <span :class="conclusion.escalation_risk > 0.6 ? 'danger' : 'normal'">
            {{ (conclusion.escalation_risk * 100).toFixed(0) }}%
          </span>
        </div>
        <div class="description">
          {{ conclusion.risk_desc }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import * as echarts from 'echarts'
import { getTopicPrediction, getUserTrajectories } from '@/api/prediction'

const props = defineProps({
  topicId: String,
  days: {
    type: Number,
    default: 3
  }
})

const predictionData = ref(null)
const anomalies = ref([])
const userTrajectories = ref([])
const conclusion = ref({})

onMounted(async () => {
  // 获取预测数据
  const response = await getTopicPrediction(props.topicId, props.days)
  predictionData.value = response.data
  
  // 获取用户轨迹
  const trajResponse = await getUserTrajectories(props.topicId, 10)
  userTrajectories.value = trajResponse.data.trajectories
  
  // 提取异常和结论
  anomalies.value = response.data.anomalies || []
  conclusion.value = response.data.conclusion
  
  // 绘制图表
  drawPredictionChart()
})

const drawPredictionChart = () => {
  const chart = echarts.init(document.getElementById('predictionChart'))
  
  const histories = predictionData.value.history
  const predictions = predictionData.value.prediction
  
  const dates = [
    ...histories.map(h => h.date),
    ...predictions.map(p => p.date)
  ]
  
  const posRatios = [
    ...histories.map(h => h.pos_ratio),
    null,  // 间隔
    ...predictions.map(p => p.pos_ratio)
  ]
  
  const negRatios = [
    ...histories.map(h => h.neg_ratio),
    null,
    ...predictions.map(p => p.neg_ratio)
  ]
  
  const option = {
    tooltip: { trigger: 'axis' },
    legend: { data: ['正面', '消极'] },
    xAxis: { 
      type: 'category', 
      data: dates,
      boundaryGap: false 
    },
    yAxis: { 
      type: 'value',
      max: 1,
      axisLabel: { formatter: '{value}' }
    },
    series: [
      {
        name: '正面',
        data: posRatios,
        type: 'line',
        smooth: true,
        itemStyle: { color: '#52c41a' }
      },
      {
        name: '消极',
        data: negRatios,
        type: 'line',
        smooth: true,
        itemStyle: { color: '#f5222d' }
      }
    ]
  }
  
  chart.setOption(option)
}

const formatAlertType = (type) => {
  const typeMap = {
    'negative_sentiment_spike': '负面情感激增',
    'high_risk_user_surge': '高风险用户激增',
    'sentiment_plunge': '情感快速下降'
  }
  return typeMap[type] || type
}
</script>

<style scoped>
.prediction-container {
  padding: 20px;
}

.chart-container {
  margin-bottom: 30px;
}

#predictionChart {
  height: 400px;
}

.alerts-container {
  margin-bottom: 30px;
}

.alert-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 15px;
}

.alert-card {
  padding: 15px;
  border-left: 4px solid;
  background: #f5f5f5;
}

.alert-card.severity-5 { border-color: #f5222d; background: #fff1f0; }
.alert-card.severity-4 { border-color: #fa8c16; background: #fff7e6; }
.alert-card.severity-3 { border-color: #faad14; background: #fffbe6; }

.alert-type {
  font-weight: bold;
  margin-bottom: 8px;
}

.alert-desc {
  margin-bottom: 8px;
  font-size: 14px;
}

.alert-meta {
  font-size: 12px;
  color: #666;
  display: flex;
  gap: 15px;
}

.user-table {
  width: 100%;
  border-collapse: collapse;
}

.user-table thead {
  background: #fafafa;
}

.user-table td {
  padding: 12px;
  border-bottom: 1px solid #ddd;
}

.user-table tr.risk-level-5 { background: #fff1f0; }
.user-table tr.risk-level-4 { background: #fff7e6; }
.user-table tr.risk-level-3 { background: #fffbe6; }

.level-badge {
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: bold;
}

.level-badge.level-5 { background: #f5222d; color: white; }
.level-badge.level-4 { background: #fa8c16; color: white; }
.level-badge.level-3 { background: #faad14; color: white; }
.level-badge.level-2 { background: #1890ff; color: white; }
.level-badge.level-1 { background: #52c41a; color: white; }

.conclusion-container {
  margin-top: 30px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 4px;
}

.conclusion-card {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.metric {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.metric .label {
  font-weight: bold;
  margin-right: 10px;
}

.description {
  grid-column: 1 / -1;
  padding: 15px;
  background: white;
  border-radius: 4px;
  border-left: 3px solid #1890ff;
}
</style>
```

---

## 第七部分：实施路线图

### Phase 1：基础设施准备（1周）
- [ ] 设计和创建知识图谱数据结构
- [ ] 实现 `SentimentKnowledgeGraph` 类
- [ ] 创建测试数据集

### Phase 2：时序模块开发（1周）
- [ ] 实现 `TemporalGraphSnapshot` 类
- [ ] 实现时序快照序列管理
- [ ] 实现异常检测逻辑

### Phase 3：KGCN模型（1-2周）
- [ ] 实现 `KGCNLayer` 和 `SentimentKGCN`
- [ ] 训练/微调模型
- [ ] 模型评估和保存

### Phase 4：预测管道集成（1周）
- [ ] 实现 `TemporalKGCNPredictor`
- [ ] 集成到 FastAPI
- [ ] 创建数据库表和ORM映射

### Phase 5：定时任务和展示（1周）
- [ ] 集成 APScheduler 定时任务
- [ ] 创建前端展示组件
- [ ] 系统测试和优化

---

## 常见问题与解决方案

### Q1: KGCN 在舆情场景中的关键优势是什么？

**A**: 
1. **多源信息聚合**：整合用户、评论、实体、关键词等多种信息
2. **关系感知**：捕捉用户与话题的多层次关系
3. **个性化权重**：不同用户对同一关系有不同重视程度
4. **可解释性**：通过路径追踪理解预测的原因

### Q2: 时序信息如何在 KGCN 中建模？

**A**: 
使用多个时间切片的图快照，通过 LSTM 聚合：
- 每个快照 → KGCN 提取该时刻的实体表示
- 多个快照输出 → LSTM 学习时间依赖
- LSTM 最后隐态 → 预测未来

### Q3: 如何区分"正常波动"和"异常信号"？

**A**:
多维度异常检测：
1. **统计异常**：情感分布突变 > 阈值
2. **用户异常**：高风险用户数激增
3. **时序异常**：同一用户情感快速转负
4. **综合评分**：加权组合判定

### Q4: 冷启动问题怎么处理？

**A**:
对新话题：
- 首先使用 BERT 单独评分所有评论
- 如果评论不足（<100），使用 BERT + 简单统计
- 当评论>=100时，转向 KGCN 预测

---

## 总结

本方案的核心流程：

```
爬取评论
    ↓
情感标注 (BERT)
    ↓
构建知识图谱
    ↓
生成时序快照
    ↓
KGCN 聚合 (多个快照)
    ↓
LSTM 时序建模
    ↓
情感趋势预测 + 异常检测
    ↓
展示预警信号
    ↓
预防舆情二次发酵 ✓
```

这个方案**既保留了KGCN的优势**（多源信息融合、关系感知），**又补充了时序建模能力**（LSTM），特别适合你的**舆情预警场景**。

