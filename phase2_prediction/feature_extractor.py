"""
phase2_prediction/feature_extractor.py - 从知识图谱和时序快照中提取 KGCN 特征

输出特征类型：
  1. 节点特征向量（Node Feature Matrix）
  2. 时序特征序列（Temporal Feature Sequence，来自快照）
  3. 邻居采样特征（Neighbor Aggregation，KGCN 核心）
  4. 图级中心性特征（PageRank / Degree / Betweenness）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple

import networkx as nx

from data_models import NodeType, SentimentRange
from phase1_kg_snapshot.snapshot import GraphSnapshot, TemporalSnapshotGenerator


class FeatureExtractor:
    """
    从知识图谱和时序快照中提取特征，供 KGCN 模型使用。

    KGCN 所需输入：
      - entity_embeddings : 实体嵌入矩阵（节点 → 特征向量）
      - relation_embeddings: 关系嵌入矩阵（关系类型 → 向量）
      - interaction_pairs  : (用户, 评论/博文) 交互对 + 标签（是否负面）
      - temporal_features  : 时序快照特征序列（T × F 矩阵）
    """

    # 节点特征维度定义
    NODE_FEATURE_DIM = 16  # 每个节点的特征向量维度

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._node_index: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # 节点索引
    # ------------------------------------------------------------------

    def build_node_index(self) -> Dict[str, int]:
        """为所有节点分配整数索引（KGCN 实体 ID）。"""
        self._node_index = {nid: idx for idx, nid in enumerate(self.graph.nodes())}
        return self._node_index

    def get_node_index(self) -> Dict[str, int]:
        if self._node_index is None:
            self.build_node_index()
        return self._node_index

    # ------------------------------------------------------------------
    # 节点特征向量
    # ------------------------------------------------------------------

    def extract_node_features(self) -> Dict[str, List[float]]:
        """
        提取每个节点的特征向量（16 维）。

        维度说明：
          [0]  node_type_user    : 是否为用户节点
          [1]  node_type_comment : 是否为评论节点
          [2]  node_type_post    : 是否为博文节点
          [3]  node_type_keyword : 是否为关键词节点
          [4]  sentiment_score   : 情感分数 [0, 1]（非情感节点为 0.5）
          [5]  sentiment_polarity: 情感极性 |score - 0.5| * 2
          [6]  influence_score   : 影响力 [0, 1]（用户节点）
          [7]  followers_log     : log10(followers + 1) / 7
          [8]  is_certified      : 认证状态 0/1
          [9]  virality_score    : 传播分数 [0, 1]（博文节点）
          [10] likes_log         : log10(likes + 1) / 6
          [11] replies_log       : log10(replies + 1) / 4
          [12] tfidf_score       : TF-IDF（关键词节点）
          [13] sentiment_tendency: 词情感倾向（关键词节点）
          [14] in_degree_norm    : 归一化入度
          [15] out_degree_norm   : 归一化出度
        """
        import math

        max_degree = max(dict(self.graph.degree()).values()) if self.graph.number_of_nodes() > 0 else 1
        max_degree = max(1, max_degree)

        features: Dict[str, List[float]] = {}

        for nid, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "")
            vec = [0.0] * self.NODE_FEATURE_DIM

            # 节点类型 one-hot
            vec[0] = 1.0 if nt == NodeType.USER else 0.0
            vec[1] = 1.0 if nt == NodeType.COMMENT else 0.0
            vec[2] = 1.0 if nt == NodeType.POST else 0.0
            vec[3] = 1.0 if nt == NodeType.KEYWORD else 0.0

            # 情感
            s = float(data.get("sentiment_score", 0.5))
            vec[4] = s
            vec[5] = abs(s - 0.5) * 2.0

            # 用户属性
            vec[6] = float(data.get("influence_score", 0.0))
            followers = int(data.get("followers_count", 0))
            vec[7] = math.log10(followers + 1) / 7.0
            vec[8] = 1.0 if data.get("is_certified", False) else 0.0

            # 博文属性
            vec[9] = float(data.get("virality_score", 0.0))
            likes = int(data.get("likes_count", 0))
            vec[10] = math.log10(likes + 1) / 6.0
            replies = int(data.get("replies_count", 0))
            vec[11] = math.log10(replies + 1) / 4.0

            # 关键词属性
            vec[12] = float(data.get("tfidf_score", 0.0))
            vec[13] = float(data.get("sentiment_tendency", 0.5))

            # 度数
            vec[14] = self.graph.in_degree(nid) / max_degree
            vec[15] = self.graph.out_degree(nid) / max_degree

            features[nid] = vec

        return features

    # ------------------------------------------------------------------
    # 时序特征序列
    # ------------------------------------------------------------------

    def extract_temporal_features(
        self, snapshots: List[GraphSnapshot]
    ) -> List[Dict]:
        """
        从快照序列提取时序特征（T × F 格式，T=快照数，F=特征维度）。
        直接调用 TemporalSnapshotGenerator.extract_temporal_features()。
        """
        return TemporalSnapshotGenerator.extract_temporal_features(snapshots)

    # ------------------------------------------------------------------
    # 邻居采样（KGCN 核心）
    # ------------------------------------------------------------------

    def sample_neighbors(
        self, node_id: str, max_neighbors: int = 8
    ) -> List[Tuple[str, str, float]]:
        """
        为指定节点采样邻居。

        返回：
          [(neighbor_id, relation_type, edge_weight), ...]
          按边权重降序排列，取前 max_neighbors 个。
        """
        neighbors = []
        # 出边邻居
        for _, dst, data in self.graph.out_edges(node_id, data=True):
            neighbors.append((dst, data.get("relation", ""), data.get("weight", 1.0)))
        # 入边邻居
        for src, _, data in self.graph.in_edges(node_id, data=True):
            neighbors.append((src, data.get("relation", ""), data.get("weight", 1.0)))

        # 按权重降序，取 top-K
        neighbors.sort(key=lambda x: -x[2])
        return neighbors[:max_neighbors]

    # ------------------------------------------------------------------
    # 关系嵌入映射
    # ------------------------------------------------------------------

    def get_relation_index(self) -> Dict[str, int]:
        """
        为所有关系类型分配整数索引（KGCN relation ID）。
        """
        relations = set()
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "unknown")
            relations.add(rel)
        return {rel: idx for idx, rel in enumerate(sorted(relations))}

    # ------------------------------------------------------------------
    # 交互对（用于 KGCN 监督学习）
    # ------------------------------------------------------------------

    def build_interaction_pairs(self) -> List[Tuple[int, int, int]]:
        """
        构建 (user_idx, item_idx, label) 交互对。
        label = 1 表示用户对该内容有负面情感（供分类训练）。

        返回：
          [(user_idx, item_idx, label), ...]
        """
        node_idx = self.get_node_index()
        pairs = []

        for src, dst, data in self.graph.edges(data=True):
            relation = data.get("relation", "")
            if relation in ("writes_post", "writes_comment", "likes_post"):
                src_type = self.graph.nodes[src].get("node_type", "")
                dst_type = self.graph.nodes[dst].get("node_type", "")
                if src_type == NodeType.USER and dst_type in (
                    NodeType.POST,
                    NodeType.COMMENT,
                ):
                    sentiment = self.graph.nodes[dst].get("sentiment_score", 0.5)
                    # 仅将明确负面（< NEGATIVE_MAX）的内容标记为 label=1，中性不计入
                    label = 1 if sentiment < SentimentRange.NEGATIVE_MAX else 0
                    if src in node_idx and dst in node_idx:
                        pairs.append((node_idx[src], node_idx[dst], label))

        return pairs

    # ------------------------------------------------------------------
    # 图级中心性特征
    # ------------------------------------------------------------------

    def compute_centrality(self) -> Dict[str, Dict[str, float]]:
        """
        计算图级中心性指标：PageRank / Degree / Betweenness。

        返回：
          {
            "pagerank"    : {node_id: score},
            "in_degree"   : {node_id: score},
            "out_degree"  : {node_id: score},
            "betweenness" : {node_id: score},  (近似，k=100)
          }
        """
        pr = nx.pagerank(self.graph, alpha=0.85) if self.graph.number_of_nodes() > 1 else {}
        in_deg = dict(self.graph.in_degree())
        out_deg = dict(self.graph.out_degree())
        try:
            bw = nx.betweenness_centrality(
                self.graph, k=min(100, self.graph.number_of_nodes()), normalized=True
            )
        except Exception:
            bw = {}
        return {
            "pagerank": pr,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "betweenness": bw,
        }
