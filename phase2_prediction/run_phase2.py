"""
phase2_prediction/run_phase2.py - Phase 2 流水线入口

执行步骤：
  1. 从本地加载已保存的知识图谱和时序快照（Phase 1 输出）
  2. 获取最近 n 天快照（默认 3 天）
  3. 提取节点特征和时序特征
  4. 调用 KGCN 模型预测未来情感趋势
  5. 多维舆情评分
  6. 异常检测与预警生成
  7. 保存预测结果

用法（命令行）：
  python phase2_prediction/run_phase2.py --topic_id topic_001
  python phase2_prediction/run_phase2.py --topic_id topic_001 --forecast_days 5 --recent_days 7
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Dict, Optional

from phase1_kg_snapshot.storage import GraphStorage
from phase2_prediction.alert_system import AlertSystem
from phase2_prediction.feature_extractor import FeatureExtractor
from phase2_prediction.kgcn_model import KGCNPredictor
from phase2_prediction.loader import GraphAndSnapshotLoader
from phase2_prediction.opinion_level import OpinionLevelCalculator


class Phase2Pipeline:
    """Phase 2 主流水线：加载图与快照 → KGCN 预测 → 评分 → 预警。"""

    def __init__(
        self,
        base_dir: str = "data",
        model_path: Optional[str] = None,
    ):
        self.loader = GraphAndSnapshotLoader(base_dir=base_dir)
        self.storage = GraphStorage(base_dir=base_dir)
        self.kgcn = KGCNPredictor(model_path=model_path)
        self.alert_system = AlertSystem()

    def run(
        self,
        topic_id: str,
        version: Optional[str] = None,
        forecast_days: int = 3,
        recent_days: int = 3,
    ) -> Dict:
        """
        运行 Phase 2 流水线。

        参数：
          topic_id     : 话题 ID
          version      : 快照版本号，None 时使用最新版本
          forecast_days: 预测天数（默认 3）
          recent_days  : 使用最近多少天的快照（默认 3）

        返回：
          包含 predictions / scores / alerts 的字典
        """
        print(f"\n{'='*60}")
        print(f"[Phase 2] 话题：{topic_id}  | 预测：{forecast_days}天")
        print(f"{'='*60}")

        # 步骤1：加载图和快照
        print("[1/5] 加载知识图谱和快照...")
        graph = self.loader.load_graph(topic_id, version=version)
        print(f"      图：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 边")

        recent_snapshots = self.loader.load_recent_snapshots(
            topic_id, n_days=recent_days, version=version
        )
        print(f"      加载最近 {recent_days} 天快照：{len(recent_snapshots)} 个")

        if not recent_snapshots:
            print("⚠️  无可用快照，请先运行 Phase 1。")
            return {"error": "no_snapshots"}

        # 步骤2：特征提取
        print("[2/5] 提取节点和时序特征...")
        extractor = FeatureExtractor(graph)
        node_features = extractor.extract_node_features()
        temporal_features = extractor.extract_temporal_features(recent_snapshots)
        print(f"      节点特征：{len(node_features)} 个节点，每节点 {extractor.NODE_FEATURE_DIM} 维")
        print(f"      时序特征：{len(temporal_features)} 个时间步")

        # 步骤3：KGCN 预测
        print("[3/5] KGCN 情感趋势预测...")
        prediction = self.kgcn.predict(
            snapshots=recent_snapshots,
            forecast_days=forecast_days,
        )
        print(f"      预测情感（未来{forecast_days}天）：{prediction.predicted_sentiments}")
        print(f"      预测负面率：{[f'{r:.1%}' for r in prediction.predicted_negative_ratios]}")
        print(f"      风险等级：L{prediction.risk_level}  置信度：{prediction.confidence:.1%}")
        if prediction.secondary_fermentation_risk:
            print("      🚨 检测到二次发酵风险！")

        # 步骤4：多维舆情评分
        print("[4/5] 多维舆情评分...")
        calculator = OpinionLevelCalculator(graph=graph)
        scores = calculator.calculate_all_scores(
            snapshots=recent_snapshots,
            prediction=prediction,
        )
        for score in scores:
            print(
                f"      [{score.snapshot_id}] "
                f"综合分={score.final_score:.1f} "
                f"等级=L{score.level}（调整后L{score.adjusted_level}）"
            )

        # 步骤5：预警生成
        print("[5/5] 生成预警...")
        alerts = self.alert_system.generate_alerts(
            snapshots=recent_snapshots,
            scores=scores,
            prediction=prediction,
        )
        print(self.alert_system.format_alerts(alerts))

        # 保存结果
        result = {
            "topic_id": topic_id,
            "predicted_at": datetime.now().isoformat(),
            "version": version or self.loader.storage.get_latest_version(topic_id),
            "forecast_days": forecast_days,
            "recent_days_used": len(recent_snapshots),
            "prediction": {
                "predicted_sentiments": prediction.predicted_sentiments,
                "predicted_negative_ratios": prediction.predicted_negative_ratios,
                "risk_level": prediction.risk_level,
                "confidence": prediction.confidence,
                "secondary_fermentation_risk": prediction.secondary_fermentation_risk,
                "explanation": prediction.explanation,
            },
            "scores": [
                {
                    "snapshot_id": s.snapshot_id,
                    "final_score": s.final_score,
                    "level": s.level,
                    "adjusted_level": s.adjusted_level,
                    "level_label": s.level_label,
                }
                for s in scores
            ],
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "level": a.level,
                    "level_label": a.level_label,
                    "trigger_reason": a.trigger_reason,
                    "recommended_action": a.recommended_action,
                    "secondary_risk": a.secondary_risk,
                }
                for a in alerts
            ],
        }

        # 保存预测结果到文件
        pred_dir = os.path.join(self.storage.predictions_dir)
        os.makedirs(pred_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_path = os.path.join(pred_dir, f"{topic_id}_{ts}_prediction.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Phase 2 完成！预测结果保存至：{pred_path}")
        return result


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2：KGCN 舆情预测与预警")
    parser.add_argument("--topic_id", required=True, help="话题 ID")
    parser.add_argument("--version", default=None, help="版本号，默认使用最新")
    parser.add_argument("--forecast_days", type=int, default=3, help="预测天数")
    parser.add_argument("--recent_days", type=int, default=3, help="使用最近 N 天快照")
    parser.add_argument("--base_dir", default="data", help="数据存储根目录")
    parser.add_argument("--model_path", default=None, help="KGCN 模型权重路径")
    args = parser.parse_args()

    pipeline = Phase2Pipeline(base_dir=args.base_dir, model_path=args.model_path)
    pipeline.run(
        topic_id=args.topic_id,
        version=args.version,
        forecast_days=args.forecast_days,
        recent_days=args.recent_days,
    )


if __name__ == "__main__":
    main()
