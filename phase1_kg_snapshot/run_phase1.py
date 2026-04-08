"""
phase1_kg_snapshot/run_phase1.py - Phase 1 流水线入口

执行步骤：
  1. 从数据库/CSV 读取评论数据
  2. 数据清洗与特征提取（关键词 TF-IDF、情感分数）
  3. 构建/追加更新知识图谱
  4. 生成时序快照
  5. 本地保存（永久化图 + 带版本存档 + 快照 + 元数据）
  6. 打印执行报告

用法（命令行）：
  python phase1_kg_snapshot/run_phase1.py --topic_id topic_001 --data_path data.csv
  python phase1_kg_snapshot/run_phase1.py --topic_id topic_001 --data_path data.csv --incremental
"""

import argparse
import csv
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# 确保根目录在 PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_models import Comment, Keyword, Post, TimeSlot, User, SentimentRange
from phase1_kg_snapshot.knowledge_graph import SentimentKnowledgeGraph
from phase1_kg_snapshot.snapshot import TemporalSnapshotGenerator
from phase1_kg_snapshot.storage import GraphStorage


# ---------------------------------------------------------------------------
# 示例：关键词提取（实际应接入 jieba + TF-IDF）
# ---------------------------------------------------------------------------

def extract_keywords_simple(text: str, top_n: int = 5) -> List[str]:
    """
    简单关键词提取示意（实际应使用 jieba + sklearn TF-IDF）。
    返回按空格分词后频率最高的 top_n 词。
    """
    import re
    words = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text)
    stop_words = {"的", "了", "是", "在", "和", "也", "都", "就", "很", "个", "这", "那", "有"}
    words = [w for w in words if len(w) > 1 and w not in stop_words]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:top_n]]


def build_keyword_map(
    all_texts: List[str],
    sentiment_scores: Optional[List[float]] = None,
) -> Dict[str, Keyword]:
    """
    从文本列表构建全局关键词映射。

    参数：
      all_texts       : 所有评论/博文文本
      sentiment_scores: 对应文本的情感分数（用于计算词的情感倾向）

    返回：
      {word: Keyword} 映射
    """
    word_freq: Dict[str, int] = {}
    word_doc_count: Dict[str, int] = {}
    word_sentiment_sum: Dict[str, float] = {}

    for idx, text in enumerate(all_texts):
        words = extract_keywords_simple(text, top_n=10)
        word_set = set(words)
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        for w in word_set:
            word_doc_count[w] = word_doc_count.get(w, 0) + 1
            if sentiment_scores:
                word_sentiment_sum[w] = (
                    word_sentiment_sum.get(w, 0.0) + sentiment_scores[idx]
                )

    total_docs = len(all_texts) or 1
    keyword_map: Dict[str, Keyword] = {}
    import math

    for word, freq in word_freq.items():
        doc_count = word_doc_count.get(word, 1)
        # 简化 TF-IDF：(freq / total_tokens) * log(total_docs / doc_count)
        tfidf = (freq / sum(word_freq.values())) * math.log(total_docs / doc_count + 1)
        sentiment_avg = (
            word_sentiment_sum.get(word, 0.0) / doc_count
            if word in word_sentiment_sum
            else 0.5
        )
        kw = Keyword(
            keyword_id=word,
            word=word,
            frequency=freq,
            tfidf_score=round(tfidf, 6),
            document_count=doc_count,
            sentiment_tendency=round(sentiment_avg, 4),
        )
        kw.update_sentiment_polarity()
        keyword_map[word] = kw
    return keyword_map


# ---------------------------------------------------------------------------
# 数据加载（示例：从 CSV 加载）
# ---------------------------------------------------------------------------

def load_from_csv(path: str):
    """
    从 CSV 文件加载数据。

    期望列（可调整）：
      comment_id, author_id, post_id, content, sentiment_score,
      likes_count, replies_count, created_at,
      [parent_comment_id] (可选)

    实际使用时请替换为数据库查询逻辑。
    """
    comments: List[Comment] = []
    users_seen: Dict[str, User] = {}
    posts_seen: Dict[str, Post] = {}

    if not os.path.exists(path):
        print(f"[Phase1] 数据文件不存在：{path}，使用内置示例数据。")
        return _generate_example_data()

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                created_at = datetime.fromisoformat(row.get("created_at", ""))
            except (ValueError, KeyError):
                created_at = datetime.now()

            author_id = row.get("author_id", "u_unknown")
            post_id = row.get("post_id", "p_unknown")
            sentiment_raw = float(row.get("sentiment_score", "0.5"))
            # 若原始分数在 [-1, 1]，归一化到 [0, 1]
            if sentiment_raw < 0:
                sentiment_raw = (sentiment_raw + 1) / 2.0
            sentiment_score = max(0.0, min(1.0, sentiment_raw))

            content = row.get("content", "")
            keywords = extract_keywords_simple(content)

            comment = Comment(
                comment_id=row.get("comment_id", f"c_{len(comments)}"),
                author_id=author_id,
                post_id=post_id,
                content=content,
                content_clean=content,
                likes_count=int(row.get("likes_count", 0)),
                replies_count=int(row.get("replies_count", 0)),
                sentiment_score=sentiment_score,
                keywords=keywords,
                created_at=created_at,
                time_slot_id=created_at.strftime("%Y%m%d"),
                parent_comment_id=row.get("parent_comment_id") or None,
            )
            comments.append(comment)

            if author_id not in users_seen:
                users_seen[author_id] = User(
                    user_id=author_id,
                    username=row.get("username", author_id),
                    followers_count=int(row.get("followers_count", 0)),
                    is_certified=row.get("is_certified", "").lower() == "true",
                )

            if post_id not in posts_seen:
                posts_seen[post_id] = Post(
                    post_id=post_id,
                    author_id=author_id,
                    content="",
                    sentiment_score=sentiment_score,
                    created_at=created_at,
                    time_slot_id=created_at.strftime("%Y%m%d"),
                )

    return list(users_seen.values()), list(posts_seen.values()), comments


def _generate_example_data():
    """生成内置示例数据（无 CSV 时使用）。"""
    from datetime import timedelta

    base_time = datetime(2024, 3, 13, 9, 0)
    users = [
        User("u1", "用户A", followers_count=500, is_certified=False, account_age_days=365),
        User("u2", "用户B", followers_count=5000, is_certified=True, account_age_days=1200),
        User("u3", "用户C", followers_count=200, is_certified=False, account_age_days=180),
        User("u4", "媒体账号", followers_count=500000, is_certified=True,
             certification_type="media", account_age_days=2000),
    ]
    posts = [
        Post("p1", "u4", "新款产品质量投诉持续增加，消费者反映强烈",
             sentiment_score=0.25, created_at=base_time,
             time_slot_id="20240313", likes_count=3200, reposts_count=850),
    ]
    raw_comments = [
        ("c1", "u1", "p1", "这款产品真的很差，质量太烂了", 0.12, base_time),
        ("c2", "u2", "p1", "之前还好，这次版本差很多", 0.30, base_time + timedelta(hours=2)),
        ("c3", "u3", "p1", "遇到同样的问题，已退款", 0.20, base_time + timedelta(hours=4)),
        ("c4", "u1", "p1", "现在彻底放弃了，极度失望", 0.08, base_time + timedelta(days=1)),
        ("c5", "u2", "p1", "负面声音越来越多，厂家沉默", 0.15, base_time + timedelta(days=1, hours=3)),
        ("c6", "u4", "p1", "多方投诉，品牌危机加剧", 0.10, base_time + timedelta(days=2)),
        ("c7", "u3", "p1", "已向消协举报", 0.05, base_time + timedelta(days=2, hours=5)),
        ("c8", "u1", "p1", "全网扩散，必须重视", 0.10, base_time + timedelta(days=2, hours=8)),
    ]
    comments = [
        Comment(
            comment_id=cid, author_id=uid, post_id=pid,
            content=text, content_clean=text,
            sentiment_score=score,
            keywords=extract_keywords_simple(text),
            created_at=ts,
            time_slot_id=ts.strftime("%Y%m%d"),
        )
        for cid, uid, pid, text, score, ts in raw_comments
    ]
    return users, posts, comments


# ---------------------------------------------------------------------------
# Phase 1 流水线
# ---------------------------------------------------------------------------

class Phase1Pipeline:
    """Phase 1 主流水线：图构建 + 快照生成 + 本地保存。"""

    def __init__(self, base_dir: str = "data"):
        self.storage = GraphStorage(base_dir=base_dir)
        self.snapshot_gen = TemporalSnapshotGenerator(granularity="day")

    def run(
        self,
        topic_id: str,
        data_path: Optional[str] = None,
        incremental: bool = False,
    ) -> Dict:
        """
        运行 Phase 1 流水线。

        参数：
          topic_id    : 话题 ID
          data_path   : 数据文件路径（CSV），None 时使用示例数据
          incremental : 若为 True，加载已有永久化图并追加新数据

        返回：
          包含 graph_summary / snapshot_count / version / 文件路径 的字典
        """
        print(f"\n{'='*60}")
        print(f"[Phase 1] 话题：{topic_id}  | 增量模式：{incremental}")
        print(f"{'='*60}")

        # 步骤1：加载数据
        print("[1/5] 加载数据...")
        if data_path:
            result = load_from_csv(data_path)
        else:
            result = _generate_example_data()
        users, posts, comments = result
        print(f"      用户：{len(users)}  博文：{len(posts)}  评论：{len(comments)}")

        # 步骤2：构建关键词映射
        print("[2/5] 提取关键词...")
        all_texts = [c.content_clean for c in comments] + [p.content for p in posts]
        sentiment_scores = [c.sentiment_score for c in comments] + [
            p.sentiment_score for p in posts
        ]
        keyword_map = build_keyword_map(all_texts, sentiment_scores)
        print(f"      关键词总数：{len(keyword_map)}")

        # 步骤3：构建时间槽
        if comments:
            timestamps = [c.created_at for c in comments if c.created_at]
            start_time = min(timestamps)
            end_time = max(timestamps)
        else:
            start_time = end_time = datetime.now()

        # 步骤4：构建/追加知识图谱
        print("[3/5] 构建知识图谱...")
        if incremental:
            existing_graph = self.storage.load_or_create_persistent_graph(topic_id)
            kg = SentimentKnowledgeGraph()
            kg.graph = existing_graph
            print(f"      加载已有图：{existing_graph.number_of_nodes()} 节点，"
                  f"{existing_graph.number_of_edges()} 边")
        else:
            kg = SentimentKnowledgeGraph()

        kg.build_from_data(
            users=users,
            posts=posts,
            comments=comments,
            keywords=keyword_map,
            time_slots=[],  # 时间槽节点由 snapshot 阶段添加
            user_map={u.user_id: u for u in users},
        )

        graph_summary = kg.summary()
        print(f"      图摘要：{graph_summary}")

        # 步骤5：生成时序快照
        print("[4/5] 生成时序快照...")
        snapshots = self.snapshot_gen.generate(
            full_graph=kg.graph,
            topic_id=topic_id,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"      快照数量：{len(snapshots)}")
        for snap in snapshots:
            print(
                f"      [{snap.start_time.date()}] "
                f"评论={snap.num_comments} "
                f"情感均值={snap.avg_sentiment:.3f} "
                f"负面={snap.negative_ratio:.1%} "
                f"{'⚠️ 异常' if snap.has_anomaly else '正常'}"
            )

        # 步骤6：保存
        print("[5/5] 保存到本地...")
        version = GraphStorage.make_version()

        # 永久化图（追加）
        self.storage.save_persistent_graph(kg.graph, topic_id)

        # 带版本的存档
        saved_paths = self.storage.save_graph(
            kg.graph, topic_id, version, formats=("pkl", "json")
        )

        # 快照
        snap_path = self.storage.save_snapshots(snapshots, topic_id, version)

        # 元数据
        self.storage.save_metadata(
            topic_id=topic_id,
            version=version,
            graph_summary=graph_summary,
            snapshot_count=len(snapshots),
        )

        result_info = {
            "topic_id": topic_id,
            "version": version,
            "graph_summary": graph_summary,
            "snapshot_count": len(snapshots),
            "saved_graph_paths": saved_paths,
            "saved_snapshot_path": snap_path,
        }

        print(f"\n✅ Phase 1 完成！版本：{version}")
        print(f"   图保存：{saved_paths}")
        print(f"   快照保存：{snap_path}")
        return result_info


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1：知识图谱与时序快照构建")
    parser.add_argument("--topic_id", required=True, help="话题 ID")
    parser.add_argument("--data_path", default=None, help="数据文件路径（CSV）")
    parser.add_argument(
        "--incremental", action="store_true", help="增量模式：追加到已有图"
    )
    parser.add_argument("--base_dir", default="data", help="数据存储根目录")
    args = parser.parse_args()

    pipeline = Phase1Pipeline(base_dir=args.base_dir)
    pipeline.run(
        topic_id=args.topic_id,
        data_path=args.data_path,
        incremental=args.incremental,
    )


if __name__ == "__main__":
    main()
