"""
phase1_kg_snapshot/knowledge_graph.py - 舆情知识图谱构建器

改进设计（对应问题1-6）：
  1. 粉丝量/点赞量/回复量 → 节点属性，不再单独建节点
  2. 添加 TimeSlot 节点支持时间维度
  3. 情感分数使用连续 [0, 1]，由 SentimentRange 区间推导类别
  4. 添加 Keyword 节点及关系，支持博文和评论
  5. 完整的 User/Post/Comment 属性系统
  6. Community 节点支持非连通图检测，提供完整图构建示例

边权重计算公式（problem 1）：
  - writes_post / writes_comment : weight = user.influence_score
  - replies_to                   : weight = comment.sentiment_score 的极性强度
  - post_has_kw / comment_has_kw : weight = keyword.tfidf_score * keyword_freq_in_doc
  - follows_in_time              : weight = |s(t+1) - s(t)| （情感变化幅度）
  - sentiment_trajectory         : weight = 情感变化绝对值
  - likes_post                   : weight = log10(likes + 1) / 5
  - belongs_to_community         : weight = community.density
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx

from data_models import (
    Comment,
    Community,
    Keyword,
    NodeType,
    Post,
    RelationType,
    SentimentRange,
    TimeSlot,
    User,
)


class SentimentKnowledgeGraph:
    """
    舆情知识图谱构建器

    图类型：有向加权图（nx.DiGraph）
    节点属性：node_type + 对应数据模型的所有字段
    边属性  ：relation_type + weight + timestamp
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _set_node(self, node_id: str, node_type: str, **attrs):
        """添加或更新节点，合并属性。"""
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(attrs)
        else:
            self.graph.add_node(node_id, node_type=node_type, **attrs)

    def _set_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        weight: float = 1.0,
        timestamp: Optional[datetime] = None,
        **attrs,
    ):
        """添加或更新有向边。"""
        self.graph.add_edge(
            src,
            dst,
            relation=relation,
            weight=round(weight, 6),
            timestamp=timestamp,
            **attrs,
        )

    @staticmethod
    def _influence_weight(user: User) -> float:
        """用户影响力边权重：先确保已计算 influence_score。"""
        if user.influence_score == 0.0:
            user.compute_influence_score()
        return max(0.01, user.influence_score)

    @staticmethod
    def _sentiment_polarity(score: float) -> float:
        """情感极性强度 = 偏离中性的程度，∈ [0, 1]。"""
        return abs(score - 0.5) * 2.0

    # ------------------------------------------------------------------
    # 添加节点
    # ------------------------------------------------------------------

    def add_user_node(self, user: User):
        """添加用户节点（粉丝量/认证状态均为属性）。"""
        user.compute_influence_score()
        self._set_node(
            f"user_{user.user_id}",
            NodeType.USER,
            user_id=user.user_id,
            username=user.username,
            followers_count=user.followers_count,
            following_count=user.following_count,
            is_certified=user.is_certified,
            certification_type=user.certification_type,
            account_age_days=user.account_age_days,
            total_posts=user.total_posts,
            avg_engagement=user.avg_engagement,
            reputation_score=user.reputation_score,
            influence_score=user.influence_score,
            registered_at=user.registered_at,
            last_active_at=user.last_active_at,
        )

    def add_post_node(self, post: Post):
        """添加博文节点（点赞/转发/回复均为属性）。"""
        post.compute_virality_score()
        self._set_node(
            f"post_{post.post_id}",
            NodeType.POST,
            post_id=post.post_id,
            author_id=post.author_id,
            content_clean=post.content_clean,
            likes_count=post.likes_count,
            reposts_count=post.reposts_count,
            comments_count=post.comments_count,
            spread_coefficient=post.spread_coefficient,
            virality_score=post.virality_score,
            sentiment_score=post.sentiment_score,
            created_at=post.created_at,
            time_slot_id=post.time_slot_id,
        )

    def add_comment_node(self, comment: Comment):
        """添加评论节点（点赞/回复量均为属性）。"""
        self._set_node(
            f"comment_{comment.comment_id}",
            NodeType.COMMENT,
            comment_id=comment.comment_id,
            author_id=comment.author_id,
            post_id=comment.post_id,
            content_clean=comment.content_clean,
            likes_count=comment.likes_count,
            replies_count=comment.replies_count,
            sentiment_score=comment.sentiment_score,
            sentiment_label=SentimentRange.label(comment.sentiment_score),
            created_at=comment.created_at,
            time_slot_id=comment.time_slot_id,
            parent_comment_id=comment.parent_comment_id,
        )

    def add_keyword_node(self, keyword: Keyword):
        """添加关键词节点（词频/TF-IDF/情感倾向均为属性）。"""
        keyword.update_sentiment_polarity()
        self._set_node(
            f"kw_{keyword.keyword_id}",
            NodeType.KEYWORD,
            keyword_id=keyword.keyword_id,
            word=keyword.word,
            frequency=keyword.frequency,
            tfidf_score=keyword.tfidf_score,
            document_count=keyword.document_count,
            sentiment_tendency=keyword.sentiment_tendency,
            sentiment_polarity=keyword.sentiment_polarity,
        )

    def add_time_slot_node(self, ts: TimeSlot):
        """添加时间槽节点。"""
        self._set_node(
            f"ts_{ts.slot_id}",
            NodeType.TIME_SLOT,
            slot_id=ts.slot_id,
            start_time=ts.start_time,
            end_time=ts.end_time,
            granularity=ts.granularity,
            total_comments=ts.total_comments,
            total_users=ts.total_users,
            avg_sentiment=ts.avg_sentiment,
            negative_ratio=ts.negative_ratio,
            extreme_negative_ratio=ts.extreme_negative_ratio,
        )

    def add_community_node(self, community: Community):
        """添加社群节点（用于非连通图检测）。"""
        self._set_node(
            f"comm_{community.community_id}",
            NodeType.COMMUNITY,
            community_id=community.community_id,
            label=community.label,
            community_size=community.community_size,
            internal_edges=community.internal_edges,
            external_edges=community.external_edges,
            density=community.density,
            avg_sentiment=community.avg_sentiment,
            sentiment_homogeneity=community.sentiment_homogeneity,
            total_followers=community.total_followers,
            has_kol=community.has_kol,
            detected_at=community.detected_at,
        )

    # ------------------------------------------------------------------
    # 添加关系边（含权重计算公式）
    # ------------------------------------------------------------------

    def connect_user_writes_post(self, user: User, post: Post):
        """
        用户 → 发布 → 博文
        边权重 = user.influence_score（影响力越高的用户，发文影响越大）
        """
        self._set_edge(
            f"user_{user.user_id}",
            f"post_{post.post_id}",
            RelationType.USER_WRITES_POST,
            weight=self._influence_weight(user),
            timestamp=post.created_at,
        )

    def connect_user_writes_comment(self, user: User, comment: Comment):
        """
        用户 → 发表 → 评论
        边权重 = user.influence_score
        """
        self._set_edge(
            f"user_{user.user_id}",
            f"comment_{comment.comment_id}",
            RelationType.USER_WRITES_COMMENT,
            weight=self._influence_weight(user),
            timestamp=comment.created_at,
        )

    def connect_user_likes_post(self, user: User, post: Post):
        """
        用户 → 点赞 → 博文
        边权重 = log10(post.likes_count + 1) / 5（归一化到 ~0-1）
        """
        weight = math.log10(post.likes_count + 1) / 5.0
        self._set_edge(
            f"user_{user.user_id}",
            f"post_{post.post_id}",
            RelationType.USER_LIKES_POST,
            weight=min(1.0, weight),
            timestamp=None,
        )

    def connect_user_follows(self, follower: User, followee: User):
        """
        用户 → 关注 → 用户
        边权重 = followee.influence_score（被关注者越有影响力，关系越重要）
        """
        self._set_edge(
            f"user_{follower.user_id}",
            f"user_{followee.user_id}",
            RelationType.USER_FOLLOWS,
            weight=self._influence_weight(followee),
        )

    def connect_comment_replies_to(
        self,
        child: Comment,
        parent_comment_id: Optional[str] = None,
        parent_post_id: Optional[str] = None,
    ):
        """
        评论 → 回复 → 评论 / 博文
        边权重 = 情感极性强度（情感越极端，回复越值得关注）
        """
        weight = self._sentiment_polarity(child.sentiment_score)
        if parent_comment_id:
            self._set_edge(
                f"comment_{child.comment_id}",
                f"comment_{parent_comment_id}",
                RelationType.COMMENT_REPLIES_TO,
                weight=max(0.01, weight),
                timestamp=child.created_at,
            )
        elif parent_post_id:
            self._set_edge(
                f"comment_{child.comment_id}",
                f"post_{parent_post_id}",
                RelationType.COMMENT_REPLIES_TO,
                weight=max(0.01, weight),
                timestamp=child.created_at,
            )

    def connect_post_keyword(self, post: Post, keyword: Keyword, freq_in_doc: int = 1):
        """
        博文 → 包含 → 关键词（问题4）
        边权重 = keyword.tfidf_score * freq_in_doc（词频+TF-IDF 综合）
        """
        weight = keyword.tfidf_score * freq_in_doc
        self._set_edge(
            f"post_{post.post_id}",
            f"kw_{keyword.keyword_id}",
            RelationType.POST_CONTAINS_KEYWORD,
            weight=max(0.01, min(1.0, weight)),
        )

    def connect_comment_keyword(
        self, comment: Comment, keyword: Keyword, freq_in_doc: int = 1
    ):
        """
        评论 → 包含 → 关键词（问题4）
        边权重 = keyword.tfidf_score * freq_in_doc
        """
        weight = keyword.tfidf_score * freq_in_doc
        self._set_edge(
            f"comment_{comment.comment_id}",
            f"kw_{keyword.keyword_id}",
            RelationType.COMMENT_CONTAINS_KEYWORD,
            weight=max(0.01, min(1.0, weight)),
        )

    def connect_node_to_time_slot(
        self, node_id: str, time_slot: TimeSlot, timestamp: Optional[datetime] = None
    ):
        """
        节点（评论/博文） → 发生于 → 时间槽
        边权重 = 1.0（时间隶属关系，无需加权）
        """
        self._set_edge(
            node_id,
            f"ts_{time_slot.slot_id}",
            RelationType.AT_TIME_SLOT,
            weight=1.0,
            timestamp=timestamp,
        )

    def connect_temporal_comments(
        self, earlier: Comment, later: Comment
    ):
        """
        评论1（早） → 时序后 → 评论2（晚）
        边权重 = |later.sentiment - earlier.sentiment|（情感变化幅度）
        此关系是 KGCN 时序建模的核心！
        """
        delta = abs(later.sentiment_score - earlier.sentiment_score)
        self._set_edge(
            f"comment_{earlier.comment_id}",
            f"comment_{later.comment_id}",
            RelationType.FOLLOWS_IN_TIME,
            weight=max(0.01, delta),
            timestamp=later.created_at,
            sentiment_delta=round(later.sentiment_score - earlier.sentiment_score, 4),
        )

    def connect_user_sentiment_trajectory(self, user_comments: List[Comment]):
        """
        同一用户的评论按时间排序后连接情感轨迹边。
        边权重 = 情感变化绝对值（变化越大，轨迹越异常）
        这是检测"情感轨迹突变"（二次发酵信号）的核心关系。
        """
        sorted_comments = sorted(
            user_comments, key=lambda c: c.created_at or datetime.min
        )
        for i in range(len(sorted_comments) - 1):
            prev = sorted_comments[i]
            curr = sorted_comments[i + 1]
            delta = abs(curr.sentiment_score - prev.sentiment_score)
            self._set_edge(
                f"comment_{prev.comment_id}",
                f"comment_{curr.comment_id}",
                RelationType.USER_SENTIMENT_TRAJECTORY,
                weight=max(0.01, delta),
                timestamp=curr.created_at,
                user_id=prev.author_id,
                sentiment_delta=round(
                    curr.sentiment_score - prev.sentiment_score, 4
                ),
            )

    def connect_node_to_community(
        self, node_id: str, community: Community
    ):
        """
        节点 → 属于 → 社群（问题6）
        边权重 = community.density（社群密度越高，归属越强）
        """
        self._set_edge(
            node_id,
            f"comm_{community.community_id}",
            RelationType.BELONGS_TO_COMMUNITY,
            weight=max(0.01, community.density),
        )

    # ------------------------------------------------------------------
    # 社群检测（问题6 - 非连通图支持）
    # ------------------------------------------------------------------

    def detect_communities(self) -> List[Community]:
        """
        非连通图检测：识别所有连通分量作为社群。

        说明：
          若图中存在多个不相连的子图（如两个互不相关的话题），
          每个连通分量被识别为一个 Community 节点。

        更高级的社群检测可使用 Louvain 算法（需安装 python-louvain），
        此处使用内置的弱连通分量检测作为基础实现。
        """
        communities: List[Community] = []
        undirected = self.graph.to_undirected()

        for idx, component in enumerate(nx.connected_components(undirected)):
            subgraph = self.graph.subgraph(component)

            # 计算内/外边数
            internal = subgraph.number_of_edges()
            n = len(component)
            # 使用无向子图计算密度（undirected: 最大边数 = n*(n-1)/2）
            undirected_sub = subgraph.to_undirected()
            density = (
                undirected_sub.number_of_edges() / (n * (n - 1) / 2) if n > 1 else 0.0
            )

            # 计算社群平均情感
            sentiment_scores = [
                self.graph.nodes[nid].get("sentiment_score", 0.5)
                for nid in component
                if "sentiment_score" in self.graph.nodes[nid]
            ]
            avg_sent = (
                sum(sentiment_scores) / len(sentiment_scores)
                if sentiment_scores
                else 0.5
            )

            # 是否包含 KOL
            has_kol = any(
                self.graph.nodes[nid].get("is_certified", False)
                or self.graph.nodes[nid].get("followers_count", 0) > 100_000
                for nid in component
            )

            # 总粉丝数
            total_followers = sum(
                self.graph.nodes[nid].get("followers_count", 0)
                for nid in component
            )

            comm = Community(
                community_id=f"comm_{idx}",
                community_size=n,
                internal_edges=internal,
                density=round(density, 4),
                avg_sentiment=round(avg_sent, 4),
                has_kol=has_kol,
                total_followers=total_followers,
                detected_at=datetime.now(),
            )
            communities.append(comm)

        return communities

    def build_communities_in_graph(self):
        """检测社群并将 Community 节点/边添加到图中。"""
        communities = self.detect_communities()
        undirected = self.graph.to_undirected()

        for idx, component in enumerate(nx.connected_components(undirected)):
            comm = communities[idx]
            self.add_community_node(comm)
            for node_id in component:
                node_type = self.graph.nodes[node_id].get("node_type", "")
                # 只将 User/Post/Comment 与社群相连，避免循环
                if node_type in (NodeType.USER, NodeType.POST, NodeType.COMMENT):
                    self.connect_node_to_community(node_id, comm)

        return communities

    # ------------------------------------------------------------------
    # 完整图构建入口
    # ------------------------------------------------------------------

    def build_from_data(
        self,
        users: List[User],
        posts: List[Post],
        comments: List[Comment],
        keywords: Dict[str, Keyword],
        time_slots: List[TimeSlot],
        user_map: Optional[Dict[str, User]] = None,
    ):
        """
        从原始数据批量构建完整知识图谱。

        参数：
          users      : 用户列表
          posts      : 博文列表
          comments   : 评论列表
          keywords   : {word: Keyword} 映射
          time_slots : 时间槽列表
          user_map   : {user_id: User} 映射（可选，加速查找）

        完整图构建流程：
          1. 添加时间槽节点
          2. 添加关键词节点
          3. 添加用户节点
          4. 添加博文节点 + 用户-博文边 + 博文-关键词边 + 博文-时间槽边
          5. 添加评论节点 + 用户-评论边 + 评论-关键词边 + 评论-时间槽边
          6. 添加评论回复边
          7. 添加时序关系边（全局 + 用户维度）
          8. 社群检测（非连通图支持）
        """
        if user_map is None:
            user_map = {u.user_id: u for u in users}

        # 步骤1：时间槽
        ts_map: Dict[str, TimeSlot] = {}
        for ts in time_slots:
            self.add_time_slot_node(ts)
            ts_map[ts.slot_id] = ts

        # 步骤2：关键词
        for kw in keywords.values():
            self.add_keyword_node(kw)

        # 步骤3：用户
        for user in users:
            self.add_user_node(user)

        # 步骤4：博文
        for post in posts:
            self.add_post_node(post)
            # 用户-博文
            if post.author_id in user_map:
                self.connect_user_writes_post(user_map[post.author_id], post)
            # 博文-关键词
            for kw_word in post.keywords:
                if kw_word in keywords:
                    self.connect_post_keyword(post, keywords[kw_word])
            # 博文-时间槽
            if post.time_slot_id and post.time_slot_id in ts_map:
                self.connect_node_to_time_slot(
                    f"post_{post.post_id}",
                    ts_map[post.time_slot_id],
                    post.created_at,
                )

        # 步骤5：评论
        user_comments_map: Dict[str, List[Comment]] = {}
        for comment in comments:
            self.add_comment_node(comment)
            user_comments_map.setdefault(comment.author_id, []).append(comment)

            # 用户-评论
            if comment.author_id in user_map:
                self.connect_user_writes_comment(user_map[comment.author_id], comment)
            # 评论-关键词
            for kw_word in comment.keywords:
                if kw_word in keywords:
                    self.connect_comment_keyword(comment, keywords[kw_word])
            # 评论-时间槽
            if comment.time_slot_id and comment.time_slot_id in ts_map:
                self.connect_node_to_time_slot(
                    f"comment_{comment.comment_id}",
                    ts_map[comment.time_slot_id],
                    comment.created_at,
                )
            # 评论回复链
            if comment.parent_comment_id:
                self.connect_comment_replies_to(
                    comment, parent_comment_id=comment.parent_comment_id
                )
            else:
                self.connect_comment_replies_to(
                    comment, parent_post_id=comment.post_id
                )

        # 步骤6：全局时序关系
        sorted_all_comments = sorted(
            comments, key=lambda c: c.created_at or datetime.min
        )
        for i in range(len(sorted_all_comments) - 1):
            self.connect_temporal_comments(
                sorted_all_comments[i], sorted_all_comments[i + 1]
            )

        # 步骤7：用户情感轨迹（每个用户独立连接）
        for uid, ucomments in user_comments_map.items():
            if len(ucomments) > 1:
                self.connect_user_sentiment_trajectory(ucomments)

        # 步骤8：社群检测
        self.build_communities_in_graph()

        return self.graph

    # ------------------------------------------------------------------
    # 图摘要
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """返回图的统计摘要。"""
        node_type_counts: Dict[str, int] = {}
        relation_counts: Dict[str, int] = {}

        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

        for _, _, data in self.graph.edges(data=True):
            rt = data.get("relation", "unknown")
            relation_counts[rt] = relation_counts.get(rt, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_type_counts": node_type_counts,
            "relation_counts": relation_counts,
            "is_weakly_connected": nx.is_weakly_connected(self.graph)
            if self.graph.number_of_nodes() > 0
            else True,
            "number_of_weakly_connected_components": nx.number_weakly_connected_components(
                self.graph
            )
            if self.graph.number_of_nodes() > 0
            else 0,
        }
