"""
phase1_kg_snapshot/storage.py - 知识图谱与时序快照的本地持久化管理

功能：
  - 支持三种格式保存图：pickle（快速读写）/ GraphML（通用）/ JSON（可读）
  - 支持保存/加载时序快照序列
  - 支持元数据记录（版本、时间戳、统计信息）
  - 支持永久化知识图谱（追加更新模式，防止二次发酵监控丢失历史）

目录结构：
  data/
  ├── graphs/
  │   ├── {topic_id}_{version}.pkl
  │   ├── {topic_id}_{version}.graphml
  │   ├── {topic_id}_{version}.json
  │   └── {topic_id}_metadata.json
  ├── snapshots/
  │   ├── {topic_id}_{version}_snapshots.pkl
  │   └── {topic_id}_{version}_snapshots.json
  └── predictions/        (Phase 2 输出)
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx

from phase1_kg_snapshot.snapshot import GraphSnapshot


class GraphStorage:
    """
    知识图谱与时序快照本地存储管理

    使用场景：
      永久化知识图谱 → 每次 Phase 1 运行追加新数据到已有图
      按版本存档       → 每个版本对应一次完整快照，供历史对比
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.graphs_dir = os.path.join(base_dir, "graphs")
        self.snapshots_dir = os.path.join(base_dir, "snapshots")
        self.predictions_dir = os.path.join(base_dir, "predictions")
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in (self.graphs_dir, self.snapshots_dir, self.predictions_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # 版本号工具
    # ------------------------------------------------------------------

    @staticmethod
    def make_version(dt: Optional[datetime] = None) -> str:
        """生成版本号字符串，例如 '20240315_120000'。"""
        if dt is None:
            dt = datetime.now()
        return dt.strftime("%Y%m%d_%H%M%S")

    def _graph_path(self, topic_id: str, version: str, fmt: str) -> str:
        return os.path.join(self.graphs_dir, f"{topic_id}_{version}.{fmt}")

    def _snapshots_path(self, topic_id: str, version: str, fmt: str) -> str:
        return os.path.join(
            self.snapshots_dir, f"{topic_id}_{version}_snapshots.{fmt}"
        )

    def _metadata_path(self, topic_id: str) -> str:
        return os.path.join(self.graphs_dir, f"{topic_id}_metadata.json")

    # ------------------------------------------------------------------
    # 图的保存（多格式）
    # ------------------------------------------------------------------

    def save_graph(
        self,
        graph: nx.DiGraph,
        topic_id: str,
        version: str,
        formats: tuple = ("pkl", "json"),
    ):
        """
        保存知识图谱到本地文件。

        参数：
          graph    : nx.DiGraph
          topic_id : 话题 ID
          version  : 版本号（建议使用 make_version()）
          formats  : 保存格式列表，可选 "pkl" / "graphml" / "json"

        注意：graphml 格式不支持 datetime 等 Python 对象属性，
              会自动将不可序列化的属性转换为字符串后保存。
        """
        saved = {}

        if "pkl" in formats:
            path = self._graph_path(topic_id, version, "pkl")
            with open(path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            saved["pkl"] = path

        if "graphml" in formats:
            path = self._graph_path(topic_id, version, "graphml")
            # GraphML 不支持复杂类型，需预处理
            g_copy = nx.DiGraph()
            for nid, data in graph.nodes(data=True):
                clean = {k: self._to_str(v) for k, v in data.items()}
                g_copy.add_node(nid, **clean)
            for u, v, data in graph.edges(data=True):
                clean = {k: self._to_str(v) for k, v in data.items()}
                g_copy.add_edge(u, v, **clean)
            nx.write_graphml(g_copy, path)
            saved["graphml"] = path

        if "json" in formats:
            path = self._graph_path(topic_id, version, "json")
            graph_dict = nx.node_link_data(graph)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(graph_dict, f, ensure_ascii=False, default=self._json_default)
            saved["json"] = path

        return saved

    @staticmethod
    def _to_str(value: Any) -> str:
        """将不可序列化的值转为字符串。"""
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value) if value is not None else ""

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    # ------------------------------------------------------------------
    # 图的加载
    # ------------------------------------------------------------------

    def load_graph(
        self, topic_id: str, version: str, fmt: str = "pkl"
    ) -> nx.DiGraph:
        """
        加载已保存的知识图谱。

        参数：
          fmt: "pkl"（推荐，保留所有 Python 对象） / "graphml" / "json"
        """
        path = self._graph_path(topic_id, version, fmt)
        if not os.path.exists(path):
            raise FileNotFoundError(f"图文件不存在：{path}")

        if fmt == "pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif fmt == "graphml":
            return nx.read_graphml(path)
        elif fmt == "json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return nx.node_link_graph(data)
        else:
            raise ValueError(f"不支持的格式：{fmt}")

    # ------------------------------------------------------------------
    # 永久化图（追加更新）
    # ------------------------------------------------------------------

    def load_or_create_persistent_graph(
        self, topic_id: str
    ) -> nx.DiGraph:
        """
        加载已有的永久化图，若不存在则创建空图。

        用于"持续监控"场景：
          每次 Phase 1 运行时调用此方法加载图，
          追加新节点/边后调用 save_persistent_graph() 保存。
        """
        path = self._graph_path(topic_id, "persistent", "pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return nx.DiGraph()

    def save_persistent_graph(self, graph: nx.DiGraph, topic_id: str):
        """
        保存永久化图（覆盖写入，不带版本号）。
        建议同时调用 save_graph() 存一份带版本号的存档。
        """
        path = self._graph_path(topic_id, "persistent", "pkl")
        with open(path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # 快照的保存/加载
    # ------------------------------------------------------------------

    def save_snapshots(
        self,
        snapshots: List[GraphSnapshot],
        topic_id: str,
        version: str,
        save_json: bool = True,
    ):
        """
        保存时序快照序列。

        参数：
          snapshots  : GraphSnapshot 列表
          save_json  : 是否额外保存可读的 JSON 摘要（不含 graph 对象）
        """
        # pickle：完整保存（含 graph 对象）
        pkl_path = self._snapshots_path(topic_id, version, "pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)

        if save_json:
            json_path = self._snapshots_path(topic_id, version, "json")
            summary = [
                {
                    "snapshot_id": s.snapshot_id,
                    "topic_id": s.topic_id,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat(),
                    "num_nodes": s.num_nodes,
                    "num_edges": s.num_edges,
                    "num_users": s.num_users,
                    "num_comments": s.num_comments,
                    "num_posts": s.num_posts,
                    "avg_sentiment": s.avg_sentiment,
                    "negative_ratio": s.negative_ratio,
                    "extreme_negative_ratio": s.extreme_negative_ratio,
                    "sentiment_std": s.sentiment_std,
                    "sentiment_delta": s.sentiment_delta,
                    "active_users": s.active_users,
                    "new_users": s.new_users,
                    "avg_virality": s.avg_virality,
                    "has_anomaly": s.has_anomaly,
                }
                for s in snapshots
            ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        return pkl_path

    def load_snapshots(
        self, topic_id: str, version: str
    ) -> List[GraphSnapshot]:
        """加载已保存的快照序列（从 pickle 文件）。"""
        path = self._snapshots_path(topic_id, version, "pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"快照文件不存在：{path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # 元数据管理
    # ------------------------------------------------------------------

    def save_metadata(
        self,
        topic_id: str,
        version: str,
        graph_summary: Dict,
        snapshot_count: int,
        extra: Optional[Dict] = None,
    ):
        """保存/更新话题元数据（追加版本记录）。"""
        meta_path = self._metadata_path(topic_id)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {"topic_id": topic_id, "versions": []}

        version_record = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "graph_summary": graph_summary,
            "snapshot_count": snapshot_count,
        }
        if extra:
            version_record.update(extra)

        metadata["versions"].append(version_record)
        metadata["latest_version"] = version
        metadata["last_updated"] = datetime.now().isoformat()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_metadata(self, topic_id: str) -> Dict:
        """加载话题元数据。"""
        path = self._metadata_path(topic_id)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def get_latest_version(self, topic_id: str) -> Optional[str]:
        """获取最新版本号。"""
        meta = self.load_metadata(topic_id)
        return meta.get("latest_version")

    def list_versions(self, topic_id: str) -> List[str]:
        """列出所有已保存的版本。"""
        meta = self.load_metadata(topic_id)
        return [v["version"] for v in meta.get("versions", [])]
