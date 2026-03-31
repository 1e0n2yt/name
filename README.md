# 舆情监控系统 - 知识图谱 + KGCN 两阶段架构

基于知识图谱（KG）和图卷积网络（KGCN）的舆情监控与二次发酵预测系统。

## 系统架构

```
Phase 1（离线构建）                Phase 2（在线推理）
─────────────────────             ──────────────────────────
评论/博文数据                      加载已保存的图和快照
    ↓                                   ↓
数据清洗 + 关键词提取              特征提取（节点特征 + 时序特征）
    ↓                                   ↓
知识图谱构建                       KGCN 情感趋势预测
    ↓                                   ↓
时序快照生成                       多维舆情评分（4维加权）
    ↓                                   ↓
本地永久化保存                     预警生成与输出
```

## 知识图谱设计改进

| 改进点 | 原设计 | 新设计 |
|--------|--------|--------|
| 粉丝量/点赞量/回复量 | 独立节点 | User/Post/Comment 的属性 |
| 情感分数 | 离散三分类 | 连续 [0,1] 分数（0=极端负面） |
| 内容特征 | 缺失 | Keyword 节点（TF-IDF + 情感倾向） |
| 时间维度 | 缺失 | TimeSlot 节点 + 时序快照 |
| 社群检测 | 不支持 | Community 节点 + 非连通图检测 |
| 边权重 | 未定义 | 每种关系有明确计算公式 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# Phase 1：构建知识图谱和时序快照（使用内置示例数据）
python phase1_kg_snapshot/run_phase1.py --topic_id topic_001

# Phase 1：使用真实 CSV 数据，增量追加到已有图
python phase1_kg_snapshot/run_phase1.py --topic_id topic_001 --data_path data.csv --incremental

# Phase 2：KGCN 预测（读取近3天快照，预测未来3天）
python phase2_prediction/run_phase2.py --topic_id topic_001

# Phase 2：自定义参数
python phase2_prediction/run_phase2.py --topic_id topic_001 --forecast_days 5 --recent_days 7
```

## 目录结构

```
├── data_models.py                      # 共享数据模型（User/Post/Comment/Keyword/TimeSlot/Community）
├── requirements.txt
├── phase1_kg_snapshot/
│   ├── knowledge_graph.py              # 知识图谱构建器（节点+边+权重）
│   ├── snapshot.py                     # 时序快照生成与特征提取
│   ├── storage.py                      # 多格式本地存储（pickle/GraphML/JSON）
│   └── run_phase1.py                   # Phase 1 入口
├── phase2_prediction/
│   ├── loader.py                       # 加载 Phase 1 保存的图和快照
│   ├── feature_extractor.py            # 节点/时序/邻居特征提取
│   ├── kgcn_model.py                   # KGCN 预测模型
│   ├── opinion_level.py                # 多维舆情评分（4维加权 L1-L5）
│   ├── alert_system.py                 # 预警生成（含二次发酵检测）
│   └── run_phase2.py                   # Phase 2 入口
└── data/                               # 运行后自动创建
    ├── graphs/                         # 知识图谱文件（pkl/json/graphml）
    ├── snapshots/                      # 时序快照文件
    └── predictions/                    # Phase 2 预测结果
```

## 持续监控（防止二次发酵）

```bash
# 每天定时执行 Phase 1（增量追加，永久化图）
python phase1_kg_snapshot/run_phase1.py --topic_id topic_001 --data_path today.csv --incremental

# 每天 Phase 2（读取近3天快照进行预测）
python phase2_prediction/run_phase2.py --topic_id topic_001 --recent_days 3
```
