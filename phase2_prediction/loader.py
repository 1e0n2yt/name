"""
phase2_prediction/loader.py - 从本地文件加载 Phase 1 构建的图和快照

功能：
  - 加载带版本号的知识图谱
  - 加载时序快照序列
  - 加载元数据
  - 获取"最新版本"或"指定日期范围"内的快照
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional

import networkx as nx

from phase1_kg_snapshot.snapshot import GraphSnapshot, TemporalSnapshotGenerator
from phase1_kg_snapshot.storage import GraphStorage


class GraphAndSnapshotLoader:
    """
    从本地加载 Phase 1 构建好的图和快照。

    典型用法：
      loader = GraphAndSnapshotLoader(base_dir="data")
      graph = loader.load_graph("topic_001")
      snapshots = loader.load_recent_snapshots("topic_001", n_days=3)
    """

    def __init__(self, base_dir: str = "data"):
        self.storage = GraphStorage(base_dir=base_dir)

    def load_graph(
        self,
        topic_id: str,
        version: Optional[str] = None,
        fmt: str = "pkl",
        use_persistent: bool = True,
    ) -> nx.DiGraph:
        """
        加载知识图谱。

        参数：
          topic_id       : 话题 ID
          version        : 版本号，None 时使用最新版本
          fmt            : 文件格式 "pkl" / "graphml" / "json"
          use_persistent : 若为 True，优先加载永久化图（包含全部历史数据）

        返回：
          nx.DiGraph
        """
        if use_persistent:
            try:
                return self.storage.load_or_create_persistent_graph(topic_id)
            except Exception:
                pass  # 回退到版本化加载

        if version is None:
            version = self.storage.get_latest_version(topic_id)
            if version is None:
                raise FileNotFoundError(
                    f"话题 {topic_id} 没有任何保存的版本，请先运行 Phase 1。"
                )

        return self.storage.load_graph(topic_id, version, fmt=fmt)

    def load_snapshots(
        self, topic_id: str, version: Optional[str] = None
    ) -> List[GraphSnapshot]:
        """
        加载时序快照序列。

        参数：
          version: None 时使用最新版本
        """
        if version is None:
            version = self.storage.get_latest_version(topic_id)
            if version is None:
                raise FileNotFoundError(
                    f"话题 {topic_id} 没有任何保存的版本，请先运行 Phase 1。"
                )
        return self.storage.load_snapshots(topic_id, version)

    def load_recent_snapshots(
        self, topic_id: str, n_days: int = 3, version: Optional[str] = None
    ) -> List[GraphSnapshot]:
        """
        加载最近 n_days 天的快照，供 KGCN 使用。

        参数：
          n_days: 需要加载的最近天数（默认 3 天）

        返回：
          按时间升序排列的 GraphSnapshot 列表
        """
        snapshots = self.load_snapshots(topic_id, version)
        return TemporalSnapshotGenerator.get_recent_snapshots(snapshots, n_days=n_days)

    def load_metadata(self, topic_id: str) -> dict:
        """加载话题元数据。"""
        return self.storage.load_metadata(topic_id)

    def list_available_versions(self, topic_id: str) -> List[str]:
        """列出话题所有可用版本。"""
        return self.storage.list_versions(topic_id)
