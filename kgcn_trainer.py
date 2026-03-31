"""
kgcn_trainer.py
KGCN 模型训练模块（代码2）

功能：
    - 加载多个时序快照构建训练数据集
    - 定义 KGCN 模型架构（支持多层图卷积聚合）
    - 使用时序快照数据训练模型
    - 输出未来 n 天的情感分布（积极、中性、消极占比）
    - 根据 opinion_level_calc.py 中的公式计算舆情风险等级（1-5 级）
    - 保存训练好的模型权重到本地

输入数据格式：
    - snapshot_dir: 时序快照目录
    - topic_name: 话题名称
    - epochs: 训练轮数（默认 100）
    - batch_size: 批次大小（默认 32）
    - forecast_days: 预测天数（默认 3）
    - model_save_path: 模型保存路径

输出数据格式：
    - 训练好的模型权重文件（.pth 格式）
    - 模型配置文件（.json 格式）
    - 训练日志（包含损失函数值）

遵循 PEP8 规范，random_state=42 保证可复现性。
"""

import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from kg_snapshot_builder import KGSnapshot, TemporalSnapshotManager
from opinion_level_calc import (
    OpinionLevelResult,
    calculate_opinion_level_from_distribution,
)

# 全局随机种子确保可复现性
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("使用设备：%s", DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# KGCN 模型架构
# ──────────────────────────────────────────────────────────────────────────────

class KGCNAggregator(nn.Module):
    """
    KGCN 邻域聚合层

    实现基于加权求和（sum）聚合器的图卷积操作。
    通过线性变换 + 激活函数对节点表示进行更新。

    Args:
        input_dim (int): 输入特征维度
        output_dim (int): 输出特征维度
        dropout (float): Dropout 比例，默认 0.1
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Xavier 初始化
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征张量，形状 [batch, input_dim]

        Returns:
            torch.Tensor: 输出特征张量，形状 [batch, output_dim]
        """
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        return self.layer_norm(out)


class KGCNTemporalEncoder(nn.Module):
    """
    时序特征编码器

    使用 LSTM 对时序快照特征序列进行编码，捕捉情感演变趋势。
    时序编码是 KGCN 预测情感走势的核心组件。

    Args:
        input_dim (int): 每个时间步的输入特征维度
        hidden_dim (int): LSTM 隐藏层维度
        num_layers (int): LSTM 层数，默认 2
        dropout (float): Dropout 比例，默认 0.1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): 时序输入，形状 [batch, seq_len, input_dim]

        Returns:
            Tuple[torch.Tensor, Tuple]: LSTM 输出 [batch, seq_len, hidden_dim] 及隐藏状态
        """
        return self.lstm(x)


class SentimentKGCN(nn.Module):
    """
    基于知识图谱的情感趋势预测模型（Sentiment KGCN）

    架构：
        1. 图卷积聚合层（KGCNAggregator × num_layers）
           - 对每个时序快照的图特征进行多层卷积聚合
        2. 时序编码器（LSTM）
           - 对快照序列进行时序建模，捕捉情感演变趋势
        3. 预测头（MLP）
           - 输出未来 forecast_days 天的情感分布（softmax 归一化）

    Args:
        node_feature_dim (int): 节点特征维度（来自快照特征向量）
        hidden_dim (int): 隐藏层维度，默认 64
        lstm_hidden_dim (int): LSTM 隐藏层维度，默认 64
        num_kgcn_layers (int): KGCN 卷积层数，默认 2
        lstm_layers (int): LSTM 层数，默认 2
        forecast_days (int): 预测天数，默认 3
        dropout (float): Dropout 比例，默认 0.1
    """

    def __init__(
        self,
        node_feature_dim: int = 5,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        num_kgcn_layers: int = 2,
        lstm_layers: int = 2,
        forecast_days: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.forecast_days = forecast_days

        # KGCN 卷积层序列（逐层扩展特征）
        kgcn_layers = []
        in_dim = node_feature_dim
        for i in range(num_kgcn_layers):
            out_dim = hidden_dim * (2 ** i)
            kgcn_layers.append(KGCNAggregator(in_dim, out_dim, dropout=dropout))
            in_dim = out_dim
        self.kgcn_layers = nn.ModuleList(kgcn_layers)
        self.kgcn_output_dim = in_dim

        # 时序编码器（LSTM）
        self.temporal_encoder = KGCNTemporalEncoder(
            input_dim=self.kgcn_output_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        # 预测头：[LSTM最后隐藏状态] → [forecast_days × 3]（3 = pos/neu/neg）
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_days * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 时序快照特征序列，
                              形状 [batch_size, seq_len, node_feature_dim]

        Returns:
            torch.Tensor: 预测的情感分布，
                         形状 [batch_size, forecast_days, 3]
                         其中最后一维为 [pos_ratio, neu_ratio, neg_ratio]，
                         每个时间步归一化为概率分布（softmax）
        """
        batch_size, seq_len, _ = x.shape

        # 1. KGCN 卷积聚合（对每个时间步的特征进行逐层变换）
        x_flat = x.view(batch_size * seq_len, -1)  # [batch*seq, feature_dim]
        for layer in self.kgcn_layers:
            x_flat = layer(x_flat)
        x_agg = x_flat.view(batch_size, seq_len, -1)  # [batch, seq, hidden]

        # 2. 时序编码（LSTM）
        lstm_out, _ = self.temporal_encoder(x_agg)  # [batch, seq, lstm_hidden]
        # 取最后一个时间步的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # [batch, lstm_hidden]

        # 3. 情感分布预测
        raw_output = self.predictor(last_hidden)  # [batch, forecast_days * 3]
        output = raw_output.view(batch_size, self.forecast_days, 3)  # [batch, days, 3]

        # Softmax 归一化确保每天的分布加和为 1
        output = torch.softmax(output, dim=-1)
        return output

    def get_config(self) -> Dict:
        """返回模型配置字典（用于保存）"""
        return {
            "node_feature_dim": self.node_feature_dim,
            "hidden_dim": self.hidden_dim,
            "lstm_hidden_dim": self.temporal_encoder.lstm.hidden_size,
            "num_kgcn_layers": len(self.kgcn_layers),
            "lstm_layers": self.temporal_encoder.lstm.num_layers,
            "forecast_days": self.forecast_days,
            "model_class": "SentimentKGCN",
            "version": "1.0",
        }


# ──────────────────────────────────────────────────────────────────────────────
# 训练数据集
# ──────────────────────────────────────────────────────────────────────────────

class SnapshotSequenceDataset(Dataset):
    """
    时序快照序列训练数据集

    将快照序列切分为滑动窗口，每个样本包含：
    - 输入序列：连续 seq_len 天的快照特征
    - 标签序列：后续 forecast_days 天的情感分布

    Args:
        snapshots (List[KGSnapshot]): 有序快照列表
        seq_len (int): 输入序列长度（历史天数），默认 7
        forecast_days (int): 预测天数，默认 3
    """

    def __init__(
        self,
        snapshots: List[KGSnapshot],
        seq_len: int = 7,
        forecast_days: int = 3,
    ):
        self.seq_len = seq_len
        self.forecast_days = forecast_days

        # 提取特征向量和情感分布标签
        self.features = [snap.get_feature_vector() for snap in snapshots]
        self.labels = [
            [
                snap.sentiment_distribution["positive"],
                snap.sentiment_distribution["neutral"],
                snap.sentiment_distribution["negative"],
            ]
            for snap in snapshots
        ]

        # 有效样本起始索引
        self.valid_indices = [
            i for i in range(len(self.features) - seq_len - forecast_days + 1)
        ]

        if len(self.valid_indices) == 0:
            logger.warning(
                "数据量不足以构成训练样本（共 %d 个快照，需要至少 %d 个）",
                len(snapshots),
                seq_len + forecast_days,
            )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.valid_indices[idx]
        # 输入：seq_len 天的特征序列
        x = torch.tensor(
            self.features[start:start + self.seq_len],
            dtype=torch.float32,
        )
        # 标签：后续 forecast_days 天的情感分布
        y = torch.tensor(
            self.labels[start + self.seq_len:start + self.seq_len + self.forecast_days],
            dtype=torch.float32,
        )
        return x, y


def _augment_snapshots(
    snapshots: List[KGSnapshot],
    target_count: int = 30,
    noise_std: float = 0.02,
) -> List[KGSnapshot]:
    """
    数据增强：通过添加高斯噪声扩充快照数量

    当训练数据不足时（快照数 < seq_len + forecast_days），通过在现有快照基础上
    添加轻微高斯噪声生成新快照，用于扩充训练集。

    Args:
        snapshots (List[KGSnapshot]): 原始快照列表
        target_count (int): 目标快照总数
        noise_std (float): 高斯噪声标准差，默认 0.02

    Returns:
        List[KGSnapshot]: 增强后的快照列表（原始 + 合成）
    """
    import copy

    augmented = list(snapshots)
    rng = np.random.default_rng(RANDOM_STATE)

    while len(augmented) < target_count:
        # 随机选择一个原始快照作为基础
        base_snap = snapshots[rng.integers(0, len(snapshots))]
        new_snap = copy.deepcopy(base_snap)

        # 对情感分布添加噪声并重新归一化
        dist = new_snap.sentiment_distribution
        noisy = np.array([dist["positive"], dist["neutral"], dist["negative"]])
        noisy += rng.normal(0, noise_std, size=3)
        noisy = np.clip(noisy, 0, 1)
        noisy /= noisy.sum()  # 归一化

        new_snap.sentiment_distribution = {
            "positive": float(noisy[0]),
            "neutral": float(noisy[1]),
            "negative": float(noisy[2]),
        }
        augmented.append(new_snap)

    return augmented


# ──────────────────────────────────────────────────────────────────────────────
# 模型训练器
# ──────────────────────────────────────────────────────────────────────────────

class KGCNTrainer:
    """
    KGCN 模型训练器

    负责：
    1. 加载时序快照并构建训练数据集
    2. 实例化和训练 SentimentKGCN 模型
    3. 计算训练损失并记录日志
    4. 保存模型权重（.pth）和配置（.json）到本地
    """

    def __init__(
        self,
        snapshot_dir: str = "./snapshots",
        model_save_path: str = "./models",
        seq_len: int = 7,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        num_kgcn_layers: int = 2,
        lstm_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """
        初始化训练器

        Args:
            snapshot_dir (str): 快照根目录
            model_save_path (str): 模型保存根目录
            seq_len (int): 输入序列长度（历史天数）
            hidden_dim (int): KGCN 隐藏层维度
            lstm_hidden_dim (int): LSTM 隐藏层维度
            num_kgcn_layers (int): KGCN 卷积层数
            lstm_layers (int): LSTM 层数
            learning_rate (float): 学习率
            weight_decay (float): L2 正则化系数
        """
        self.snapshot_manager = TemporalSnapshotManager(snapshot_dir=snapshot_dir)
        self.model_save_path = Path(model_save_path)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_kgcn_layers = num_kgcn_layers
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train(
        self,
        topic_name: str,
        epochs: int = 100,
        batch_size: int = 32,
        forecast_days: int = 3,
        val_split: float = 0.2,
        early_stopping_patience: int = 15,
    ) -> Dict:
        """
        训练 KGCN 模型

        Args:
            topic_name (str): 话题名称（用于加载对应快照）
            epochs (int): 训练轮数，默认 100
            batch_size (int): 批次大小，默认 32
            forecast_days (int): 预测天数，默认 3
            val_split (float): 验证集比例，默认 0.2
            early_stopping_patience (int): 早停耐心轮数，默认 15

        Returns:
            Dict: 训练结果，包含：
                - model_path (str): 模型权重保存路径
                - config_path (str): 模型配置保存路径
                - train_losses (List[float]): 训练损失序列
                - val_losses (List[float]): 验证损失序列
                - best_val_loss (float): 最佳验证损失
                - epochs_trained (int): 实际训练轮数
        """
        logger.info("开始训练话题「%s」的 KGCN 模型", topic_name)

        # 1. 加载快照
        snapshots = self.snapshot_manager.load_snapshots(topic_name)
        if not snapshots:
            raise ValueError(f"话题「{topic_name}」没有可用的快照数据")

        logger.info("加载了 %d 个快照", len(snapshots))

        # 2. 数据增强（当数据量不足时）
        min_required = self.seq_len + forecast_days + 1
        if len(snapshots) < min_required:
            target = max(min_required + 10, 30)
            logger.warning(
                "快照数量不足（%d < %d），执行数据增强至 %d 个",
                len(snapshots), min_required, target,
            )
            snapshots = _augment_snapshots(snapshots, target_count=target)

        # 3. 构建数据集
        dataset = SnapshotSequenceDataset(
            snapshots=snapshots,
            seq_len=self.seq_len,
            forecast_days=forecast_days,
        )

        if len(dataset) == 0:
            raise ValueError(
                f"数据量不足以构建训练样本，当前快照数：{len(snapshots)}，"
                f"需要至少：{self.seq_len + forecast_days}"
            )

        # 4. 划分训练集和验证集
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size

        # 按时间顺序划分（前 train_size 为训练集，后 val_size 为验证集）
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, train_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, val_size),
            shuffle=False,
        )

        logger.info("训练集：%d 样本，验证集：%d 样本", train_size, val_size)

        # 5. 实例化模型
        node_feature_dim = len(snapshots[0].get_feature_vector())
        model = SentimentKGCN(
            node_feature_dim=node_feature_dim,
            hidden_dim=self.hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            num_kgcn_layers=self.num_kgcn_layers,
            lstm_layers=self.lstm_layers,
            forecast_days=forecast_days,
        ).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("模型参数量：%d", total_params)

        # 6. 优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # 学习率调度（余弦退火）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        criterion = nn.KLDivLoss(reduction="batchmean")  # 分布之间的 KL 散度

        # 7. 训练循环
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            # ---- 训练阶段 ----
            model.train()
            train_loss_sum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                pred = model(x_batch)  # [batch, forecast_days, 3]

                # KL 散度要求输入为 log 概率，目标为概率
                pred_log = torch.log(pred + 1e-9)
                loss = criterion(
                    pred_log.view(-1, 3),
                    y_batch.view(-1, 3),
                )
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item()

            scheduler.step()
            avg_train_loss = train_loss_sum / max(1, len(train_loader))
            train_losses.append(avg_train_loss)

            # ---- 验证阶段 ----
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    pred = model(x_batch)
                    pred_log = torch.log(pred + 1e-9)
                    loss = criterion(
                        pred_log.view(-1, 3),
                        y_batch.view(-1, 3),
                    )
                    val_loss_sum += loss.item()

            avg_val_loss = val_loss_sum / max(1, len(val_loader))
            val_losses.append(avg_val_loss)

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                    epoch, epochs, avg_train_loss, avg_val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            if patience_counter >= early_stopping_patience:
                logger.info("早停触发（%d 轮无改善），在第 %d 轮停止", early_stopping_patience, epoch)
                break

        # 恢复最佳模型权重
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        logger.info(
            "训练完成！最佳验证损失=%.4f（第 %d 轮）",
            best_val_loss, best_epoch,
        )

        # 8. 保存模型
        topic_model_dir = self.model_save_path / topic_name
        topic_model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"kgcn_{topic_name}_{timestamp}.pth"
        config_filename = f"kgcn_{topic_name}_{timestamp}.json"

        model_path = topic_model_dir / model_filename
        config_path = topic_model_dir / config_filename

        # 保存模型权重
        torch.save(model.state_dict(), model_path)
        logger.info("模型权重已保存：%s", model_path)

        # 保存模型配置
        config = model.get_config()
        config.update({
            "topic_name": topic_name,
            "seq_len": self.seq_len,
            "forecast_days": forecast_days,
            "train_epochs": len(train_losses),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "saved_at": datetime.now().isoformat(),
        })
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("模型配置已保存：%s", config_path)

        return {
            "model_path": str(model_path),
            "config_path": str(config_path),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 便捷函数：预测未来情感分布
# ──────────────────────────────────────────────────────────────────────────────

def predict_future_sentiment(
    model: SentimentKGCN,
    snapshots: List[KGSnapshot],
    seq_len: int = 7,
) -> List[Dict[str, float]]:
    """
    使用训练好的模型预测未来情感分布

    Args:
        model (SentimentKGCN): 已训练的 KGCN 模型
        snapshots (List[KGSnapshot]): 历史快照列表（至少 seq_len 个）
        seq_len (int): 输入序列长度

    Returns:
        List[Dict[str, float]]: 每天的预测情感分布列表，
            例如 [{'positive': 0.2, 'neutral': 0.3, 'negative': 0.5}, ...]
    """
    if len(snapshots) < seq_len:
        raise ValueError(
            f"快照数量不足：需要至少 {seq_len} 个，当前 {len(snapshots)} 个"
        )

    # 取最近 seq_len 个快照
    recent = snapshots[-seq_len:]
    features = [snap.get_feature_vector() for snap in recent]
    x = torch.tensor([features], dtype=torch.float32).to(DEVICE)  # [1, seq_len, feature_dim]

    model.eval()
    with torch.no_grad():
        pred = model(x)  # [1, forecast_days, 3]
        pred = pred.squeeze(0).cpu().numpy()  # [forecast_days, 3]

    results = []
    for day_pred in pred:
        results.append({
            "positive": float(day_pred[0]),
            "neutral": float(day_pred[1]),
            "negative": float(day_pred[2]),
        })
    return results


def compute_risk_levels(
    predictions: List[Dict[str, float]],
    prev_snapshot: Optional[KGSnapshot] = None,
) -> List[OpinionLevelResult]:
    """
    根据预测情感分布计算舆情风险等级

    Args:
        predictions (List[Dict[str, float]]): 预测的情感分布列表
        prev_snapshot (Optional[KGSnapshot]): 最新历史快照，用于计算变化率

    Returns:
        List[OpinionLevelResult]: 每天对应的舆情等级结果
    """
    risk_levels = []
    prev_neg = (
        prev_snapshot.sentiment_distribution["negative"]
        if prev_snapshot else None
    )
    prev_count = prev_snapshot.comment_count if prev_snapshot else 0

    for pred_dist in predictions:
        result = calculate_opinion_level_from_distribution(
            sentiment_distribution=pred_dist,
            prev_neg_ratio=prev_neg,
            comment_count=prev_count,
        )
        risk_levels.append(result)
        prev_neg = pred_dist["negative"]

    return risk_levels


# ──────────────────────────────────────────────────────────────────────────────
# 便捷函数：训练入口
# ──────────────────────────────────────────────────────────────────────────────

def train_kgcn(
    topic_name: str,
    snapshot_dir: str = "./snapshots",
    model_save_path: str = "./models",
    epochs: int = 100,
    batch_size: int = 32,
    forecast_days: int = 3,
) -> Dict:
    """
    便捷函数：训练 KGCN 模型

    Args:
        topic_name (str): 话题名称
        snapshot_dir (str): 快照根目录
        model_save_path (str): 模型保存根目录
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        forecast_days (int): 预测天数

    Returns:
        Dict: 训练结果（包含模型路径、损失曲线等）

    Example:
        >>> result = train_kgcn(
        ...     topic_name="某热点话题",
        ...     snapshot_dir="./snapshots",
        ...     model_save_path="./models",
        ...     epochs=100,
        ...     forecast_days=3,
        ... )
        >>> print(f"模型已保存：{result['model_path']}")
    """
    trainer = KGCNTrainer(
        snapshot_dir=snapshot_dir,
        model_save_path=model_save_path,
    )
    return trainer.train(
        topic_name=topic_name,
        epochs=epochs,
        batch_size=batch_size,
        forecast_days=forecast_days,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 可运行示例
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import sys
    import os

    # 将父目录添加到路径（方便导入）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from kg_snapshot_builder import _generate_demo_csv, build_snapshots

    print("=" * 60)
    print("kgcn_trainer.py KGCN 模型训练示例")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = os.path.join(tmp_dir, "demo.csv")
        snap_dir = os.path.join(tmp_dir, "snapshots")
        model_dir = os.path.join(tmp_dir, "models")
        topic = "某热点话题"

        # 1. 准备数据（生成演示 CSV 并构建快照）
        print("\n[步骤1] 准备训练数据...")
        _generate_demo_csv(csv_file, num_days=14, comments_per_day=20)
        snapshots = build_snapshots(
            csv_path=csv_file,
            topic_name=topic,
            snapshot_dir=snap_dir,
            keep_days=14,
        )
        print(f"  已构建 {len(snapshots)} 个快照")

        # 2. 训练模型
        print("\n[步骤2] 训练 KGCN 模型...")
        result = train_kgcn(
            topic_name=topic,
            snapshot_dir=snap_dir,
            model_save_path=model_dir,
            epochs=50,
            batch_size=8,
            forecast_days=3,
        )
        print(f"  训练完成！实际训练轮数：{result['epochs_trained']}")
        print(f"  最佳验证损失：{result['best_val_loss']:.4f}")
        print(f"  模型路径：{result['model_path']}")
        print(f"  配置路径：{result['config_path']}")

        # 3. 使用训练好的模型进行预测
        print("\n[步骤3] 预测未来 3 天情感分布...")
        config_path = result["config_path"]
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = SentimentKGCN(
            node_feature_dim=config["node_feature_dim"],
            hidden_dim=config["hidden_dim"],
            lstm_hidden_dim=config["lstm_hidden_dim"],
            num_kgcn_layers=config["num_kgcn_layers"],
            lstm_layers=config["lstm_layers"],
            forecast_days=config["forecast_days"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(result["model_path"], map_location=DEVICE))

        future_predictions = predict_future_sentiment(model, snapshots, seq_len=7)
        risk_levels = compute_risk_levels(future_predictions, prev_snapshot=snapshots[-1])

        print(f"\n  未来 {len(future_predictions)} 天预测：")
        for i, (pred, risk) in enumerate(zip(future_predictions, risk_levels), start=1):
            print(
                f"    Day+{i}: 正={pred['positive']:.2f} 中={pred['neutral']:.2f} "
                f"负={pred['negative']:.2f}  等级={risk.level}({risk.color})"
            )

        # 显示训练损失趋势
        losses = result["train_losses"]
        print(f"\n  训练损失：初始={losses[0]:.4f}，最终={losses[-1]:.4f}")

        print("\n✅ KGCN 模型训练验证通过")
