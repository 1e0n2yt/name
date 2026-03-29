# 舆情KGCN系统 - 快速参考指南

## 核心问题回答

### Q1: 我已有BERT情感分析模型，现在想用KGCN进行预测，应该怎么做？

**A: 分5步走**

```
┌─────────────────────────────────────────┐
│ Step 1: 获取爬虫评论 (已有)             │
│ ├─ 评论ID, 用户ID, 文本, 时间戳        │
│ └─ 存入数据库                          │
└─────────┬───────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│ Step 2: BERT情感打分 (已有)            │
│ ├─ 对每条评论输入BERT                  │
│ ├─ 输出: 情感分数 (-1 to 1)            │
│ └─ 保存到数据库                        │
└─────────┬───────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│ Step 3: 构建知识图谱 (新增)           │
│ ├─ 创建节点: 评论、用户、情感、关键词  │
│ ├─ 创建边: 用户-评论、评论-情感等      │
│ └─ 重点: 添加时序关系 (评论1→评论2)   │
└─────────┬───────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│ Step 4: 生成时序快照 (新增)           │
│ ├─ 每天一个图快照                      │
│ ├─ 计算快照的情感分布                  │
│ └─ 检测异常信号                        │
└─────────┬───────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│ Step 5: KGCN预测 (新增)               │
│ ├─ 输入: 时序快照序列                  │
│ ├─ 输出: 未来情感预测                  │
│ └─ 判定: 是否有二次发酵风险            │
└─────────────────────────────────────────┘
```

### Q2: 知识图谱要怎么构建？最关键的部分是什么？

**A: 构建顺序和关键点**

```python
# 伪代码展示
kg = SentimentKnowledgeGraph()

# Step 1: 添加实体节点
kg.add_comment_node(comment)      # 评论节点
kg.add_user_node(user)            # 用户节点
kg.add_sentiment_node(label)      # 情感节点 (positive/neutral/negative)
kg.add_keyword_node(keyword)      # 关键词节点

# Step 2: 添加关系边
kg.connect_user_writes_comment(user_id, comment)  # 用户-编写-评论
kg.connect_comment_has_sentiment(comment)         # 评论-具有-情感

# Step 3: 🔑 添加时序关系 (最关键！)
for i in range(len(comments)-1):
    kg.connect_comments_temporal(comments[i], comments[i+1])

# Step 4: 用户维度时序 (第二关键！)
for user_id in users:
    user_comments = get_user_comments(user_id)
    kg.connect_user_sentiment_trajectory(user_id, user_comments)
```

**关键点**：
1. ✅ **时序关系最重要**：评论间的时间顺序连接
2. ✅ **用户维度很关键**：同一用户的评论要追踪
3. ✅ **情感强度作权重**：情感分数绝对值 = 权重
4. ❌ **不需要完全准确**：关键词和实体即使简单提取也可以

---

### Q3: 时序快照怎么理解和使用？

**A: 以天为单位的"图照片"**

```
Day 1         Day 2         Day 3         Day 4 (今天)
┌────┐       ┌────┐       ┌────┐       ┌────┐
│ G1 │       │ G2 │       │ G3 │       │ G4 │ ← 当前图
└────┘       └────┘       └────┘       └────┘
  6个评论     7个评论     5个评论      8个评论
  负面38%    负面42%     负面48%      负面56% ← 趋势！
  
快照的作用：追踪情感变化
```

**代码示例**：

```python
# 构建快照序列
sequence = TemporalGraphSequence()

for day in range(7):
    # 该天的评论
    day_comments = get_comments_by_day(day)
    
    # 构建知识图谱
    kg = SentimentKnowledgeGraph()
    graph = kg.build_from_comments(day_comments, users, topic)
    
    # 创建快照
    snapshot = TemporalGraphSnapshot(
        timestamp=day_date,
        graph=graph
    )
    
    sequence.add_snapshot(snapshot)

# 使用快照：检测异常
anomalies = sequence.detect_anomalies()
for anomaly in anomalies:
    if anomaly['type'] == 'negative_sentiment_spike':
        print(f"警告！{anomaly['description']}")
```

---

### Q4: KGCN在这里怎么工作的？为什么比单纯BERT好？

**A: KGCN做"智能聚合"**

```
BERT 只能做单条评论分析：
┌──────────┐
│ 评论文本  │ → BERT → 情感分数 ✓
└──────────┘

KGCN 能做多维关联分析：
┌──────────┐
│ 评论1    │ ───┐
├──────────┤   │
│ 评论2    │   │ → KGCN → 综合理解 ✓✓
├──────────┤   │
│ 评论3    │   │   考虑：
│(同一用户) │ ─┤   - 同一用户的情感变化趋势
├──────────┤   │   - 评论间的因果关联
│ 用户信息  │   │   - 热词和实体的情感含义
├──────────┤   │
│ 话题标签  │ ──┘
└──────────┘

优势：
1. 捕捉"情感升级"信号：用户从-0.3 → -0.6 → -0.9
2. 识别"核心触发词"：评论中的关键词是否加重负面情绪
3. 检测"传播模式"：某个用户的极端言论是否影响他人
```

**具体例子**：

```
日期    用户A的评论                    BERT分数  KGCN理解
─────────────────────────────────────────────────────
Day1   "这个产品一般"                  -0.2     中立
Day2   "没想象中好"                   -0.4     略差
Day3   "太失望了！"                   -0.7     重度失望 ← 升级！
Day4   "垃圾产品！退款！"              -0.95    极度愤怒 ← 二次发酵信号！

KGCN可以通过"用户情感轨迹"识别这个升级过程
进而判定：有舆情二次发酵风险！
```

---

### Q5: 怎样判定是否有"二次发酵"风险？

**A: 多维度异常检测**

```python
def detect_escalation_risk(sequence):
    """检测二次发酵风险"""
    
    # 指标1：负面情感比例变化
    neg_ratios = [s.sentiment_distribution['negative'] for s in sequence.snapshots]
    
    # 如果最近2天负面 > 前2天 + 15%
    if neg_ratios[-1] - neg_ratios[-3] > 0.15:
        risk_score += 0.3
        alert = "⚠️  负面情感快速上升"
    
    # 指标2：高风险用户激增
    prev_high_risk = sequence.snapshots[-2].get_high_risk_users(neg_threshold=0.6)
    curr_high_risk = sequence.snapshots[-1].get_high_risk_users(neg_threshold=0.6)
    
    if len(curr_high_risk) - len(prev_high_risk) >= 2:
        risk_score += 0.4
        alert = "⚠️  多个用户情感转极端"
    
    # 指标3：单个用户情感快速恶化
    for user_id in sequence.snapshots[-1].get_escalation_signals():
        if user.followers > 1000:  # 有影响力的用户
            risk_score += 0.3
            alert = f"⚠️  头部用户{user_id}情感恶化"
    
    # 综合判定
    if risk_score > 0.6:
        return "红色预警"  # L5
    elif risk_score > 0.4:
        return "橙色预警"  # L4
    elif risk_score > 0.2:
        return "黄色预警"  # L3
    else:
        return "正常"  # L1-2
```

**具体案例**：

```
真实舆情二次发酵场景：
─────────────────────

Day 1: 爆料某产品有质量问题
  → 负面评论 30条，负面占比 35%

Day 2: 媒体报道，更多用户跟风
  → 负面评论 150条，负面占比 52% (+17%)
  → 触发 "负面情感激增" 异常 ⚠️

Day 3: 头部博主转发，号召退款
  → 负面评论 450条，负面占比 78% (+26%)
  → 触发 "高风险用户激增" 异常 ⚠️⚠️
  → 认证用户中有5个发极端言论
  → 触发 "头部用户参与" 异常 ⚠️⚠️⚠️
  
综合判定：🔴 红色预警 (L5 - 舆情二次发酵)

如果能在 Day 2 识别到预警，可以在 Day 3 前采取行动：
✓ 发布声明澄清
✓ 媒体沟通
✓ 启动退款流程
→ 可有效控制舆论
```

---

### Q6: 集成到现有系统需要哪些修改？

**A: 最小改动方案**

```
现有系统架构：

爬虫 → 数据库 → BERT情感分析 → Web展示

新增最小化改造：

┌──────────────────────────┐
│ 原有系统                  │
│ (爬虫 + BERT)            │
└────────────┬─────────────┘
             ↓
    ┌────────────────────┐
    │ 新增模块：          │
    │ KGCN预测 (可选)    │
    │ ├─ KG构建          │
    │ ├─ 快照生成        │
    │ └─ 预测预警        │
    └────────────┬───────┘
                 ↓
    新增 API 端点：
    /api/topics/:id/prediction
    /api/topics/:id/anomalies
```

**具体步骤**：

```python
# 1. 修改数据处理流程（app/tasks/prediction_tasks.py）
@scheduler.scheduled_job('cron', hour=2, minute=0)
def daily_kgcn_prediction():
    # 在现有GNN预测后运行
    
    # 获取过去7天的评论
    comments = db.query(Comment).filter(
        Comment.created_at >= 7_days_ago
    ).all()
    
    # 新增：构建时序快照
    sequence = build_daily_snapshots(comments)
    
    # 新增：执行KGCN预测
    predictor = SentimentPredictor(model_path='model.pth')
    result = predictor.predict(sequence, forecast_days=3)
    
    # 新增：保存预测结果
    save_to_db(result)


# 2. 新增API端点（app/routes/prediction.py）
@router.get("/{topic_id}/prediction")
async def get_prediction(topic_id: str):
    # 从数据库获取已保存的预测
    result = db.query(Prediction).filter(
        Prediction.topic_id == topic_id
    ).latest()
    return result


# 3. 前端新增预测Tab（已有）
# SentimentTrendPrediction.vue
```

**数据库变化最小**：

```sql
-- 只需新增3张表（不修改现有表）

CREATE TABLE tb_prediction (
    prediction_id BIGINT PRIMARY KEY,
    topic_id VARCHAR(100),
    day1_pred_neg DECIMAL(5,4),
    day2_pred_neg DECIMAL(5,4),
    day3_pred_neg DECIMAL(5,4),
    risk_level INT,
    created_at DATETIME
);

CREATE TABLE tb_anomaly_signals (
    signal_id BIGINT PRIMARY KEY,
    topic_id VARCHAR(100),
    signal_type VARCHAR(50),
    severity DECIMAL(5,4),
    description TEXT,
    created_at DATETIME
);

-- 现有表保持不变 ✓
```

---

## 实施检查清单

### 准备阶段
- [ ] BERT模型已正常工作
- [ ] 历史评论数据>=100条/话题
- [ ] Python环境配置 (torch, networkx, jieba)
- [ ] 确认BERT输出格式 (分数范围: -1 to 1)

### 开发阶段
- [ ] 实现 `SentimentKnowledgeGraph` 类
- [ ] 实现 `TemporalGraphSnapshot` 类
- [ ] 实现 `SentimentPredictor` 类
- [ ] 训练/加载KGCN模型
- [ ] 单元测试各类功能

### 集成阶段
- [ ] 新增数据库表 (3张)
- [ ] 新增API端点 (2个)
- [ ] 新增定时任务
- [ ] 前端新增预测组件

### 上线阶段
- [ ] 7天预测准确率>70%
- [ ] 异常检测False Positive Rate<20%
- [ ] 性能测试 (10w条评论处理时间)
- [ ] 灾备方案 (KGCN失败时降级到BERT)

---

## 常见坑与解决方案

### Pit 1: 评论时间分布不均
**问题**：某天没有评论，生成的快照为空

**解决**：
```python
# 使用线性插值填充缺失的日期
def fill_missing_days(snapshots):
    all_dates = generate_date_range(start, end)
    
    for date in all_dates:
        if date not in snapshot_dict:
            # 线性插值
            prev_snapshot = get_previous(date)
            next_snapshot = get_next(date)
            
            interpolated = interpolate(prev_snapshot, next_snapshot)
            snapshots.append(interpolated)
```

### Pit 2: 小语料库过拟合
**问题**：评论数<100时，KGCN预测不准

**解决**：
```python
# 评论不足时降级到统计方法
if len(comments) < 100:
    # 使用BERT + 简单统计
    return simple_statistical_forecast(comments)
else:
    # 使用KGCN
    return kgcn_forecast(comments)
```

### Pit 3: 关键词提取不准
**问题**：jieba分词有误，垃圾关键词混入

**解决**：
```python
# 使用关键词黑名单和停用词
def extract_keywords(text):
    words = jieba.cut(text)
    
    # 过滤
    filtered = [
        w for w in words 
        if w not in STOPWORDS 
        and w not in BLACKLIST
        and len(w) > 1
        and is_chinese(w)
    ]
    
    return filtered
```

### Pit 4: 模型权重不可用
**问题**：KGCN模型没有预训练权重

**解决**：
```python
# 方案A：自己训练（需要标注数据）
# 方案B：使用简化版本（只用图结构，不用预训练权重）
# 方案C：迁移学习（从推荐系统KGCN权重迁移）

# 目前建议用 方案B：
model = SentimentKGCN(config)
# 不load权重，直接用随机初始化
# 模型会自动学习情感预测
```

---

## 性能与成本估算

### 计算复杂度

```
N = 评论数
E = 平均每条评论的邻域数

知识图谱构建：O(N*E) = O(N) (E通常很小)
时序快照：O(7*N) = O(N)
KGCN前向传播：O(N*hidden_dim^2) ≈ O(N)

总体：O(N) - 线性复杂度，性能很好 ✓

10万条评论：
- 构建KG: ~2秒
- 生成快照: ~1秒
- KGCN预测: ~3秒
- 总耗时: ~6秒 ✓
```

### 存储成本

```
图数据存储：
- 节点：N条评论 + M个用户 + 几十个关键词节点
  ≈ 100w评论 → 100w节点
  每节点 ~200字节 → ~200MB

- 边：N条评论 * 平均5条边 = 500w条边
  每条边 ~100字节 → ~500MB

总存储：~1GB（100w条评论）✓ 可接受

相比MySQL存储评论本身 (每条1KB) 的1GB：
增加存储 ~1GB，但性能提升10倍+
```

---

## 下一步建议

### 短期 (1-2周)
1. ✅ 搭建基础KGCN框架 (已有代码)
2. ✅ 用历史数据测试准确率
3. ✅ 优化超参数

### 中期 (3-4周)
1. 集成到现有系统
2. 前端展示优化
3. 用户反馈收集

### 长期 (2个月+)
1. 收集标注数据，训练专用KGCN
2. 多模态融合 (文本 + 图像 + 用户属性)
3. 跨平台舆情预测 (微博 + 抖音 + 知乎)

---

## 参考资源

- 论文原文：Knowledge Graph Convolutional Networks for Recommender Systems (WWW 2019)
- KGCN实现：<https://github.com/hwwang55/KGCN>
- 时序预测：Temporal Graph Learning (ICLR 2020)

---

## 快速Q&A

**Q: 必须用PyTorch吗？**
A: 不必，可以用TensorFlow/JAX。本指南用PyTorch只是示例。

**Q: BERT模型可以换吗？**
A: 可以。只要能输出 -1 to 1 的情感分数即可。

**Q: 能用图数据库 (Neo4j) 吗？**
A: 可以，但NetworkX已足够。如果规模>1000w条边，考虑Neo4j。

**Q: 预测准确率能到多少？**
A: 一般75-85%。具体取决于：
  - 评论数量 (越多越准)
  - 情感标签质量 (BERT准确率)
  - 历史数据长度 (>=7天)

**Q: 怎么处理多语言评论？**
A: 用多语言BERT如mBERT，其他逻辑不变。

