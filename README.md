# BiVAE-CPI：基于双向变分自编码器的化合物-蛋白质相互作用预测

## 项目简介

BiVAE-CPI 是一个基于深度学习的化合物-蛋白质相互作用（Compound-Protein Interaction, CPI）预测模型。该项目利用**双向变分自编码器（Bilateral Variational Autoencoder, BiVAE）**和**量子玻尔兹曼机变分自编码器（BiQBMVAE）**从化合物和蛋白质的交互矩阵中学习潜在表示，结合图神经网络和卷积神经网络对化合物和蛋白质进行特征提取，从而实现高精度的相互作用预测。

### 核心特性

- **双模型架构**：支持传统 BiVAE 和量子增强的 BiQBMVAE 两种模型
- **量子计算集成**：集成Kaiwu 量子计算框架，利用量子玻尔兹曼机增强特征学习
- **图神经网络**：使用 Graph Isomorphism Network (GIN) 处理化合物的分子图结构
- **端到端训练**：自动化的数据预处理、模型训练和评估流程
- **多数据集支持**：支持 Human 和 C.elegans 等数据集

---

## 环境要求
### 依赖包

```
python==3.9
numpy==1.21.5
pandas==1.2.4
rdkit==2023.3.1
scikit-learn==1.1.2
torch==1.12.1
dgl (Deep Graph Library)
networkx==3.0
kaiwu (腾讯 Kaiwu 量子计算框架)
```

### 安装步骤

1. **创建虚拟环境**（推荐）
```bash
conda create -n bivae_cpi python=3.9
conda activate bivae_cpi
```

2. **安装依赖**
```bash
pip install numpy==1.21.5 pandas==1.2.4 scikit-learn==1.1.2
pip install torch==1.12.1
pip install rdkit==2023.3.1
pip install dgl
pip install networkx==3.0
```

3. **安装 Kaiwu 量子计算框架**
```bash
# 根据项目中的 whl 文件安装
pip install kaiwu-1.3.0-cp310-none-manylinux1_x86_64.whl
```

---

## 项目结构

```
BiVAE-CPI-main/
├── code/                      # 源代码目录
│   ├── main.py               # 主训练脚本
│   ├── model.py              # 模型定义
│   ├── data_process.py       # 数据预处理脚本
│   └── utils.py              # 工具函数
├── data/                      # 原始数据目录
│   ├── human/                # Human 数据集
│   └── celegans/             # C.elegans 数据集
├── dataset/                   # 预处理后的数据集
│   ├── human/
│   │   └── origin/           # 包含处理后的 .npy 文件和字典
│   └── celegans/
├── result/                    # 训练结果保存目录
└── README.md                  # 英文说明文档
```

---

## 快速开始

### 1. 数据预处理

首先需要从原始数据生成交互矩阵和特征表示：

```bash
cd code
python data_process.py
```

**说明**：
- `data_process.py` 会读取 `../data/{dataset}/3/data.txt` 文件
- 生成的数据保存在 `../dataset/{dataset}/3/` 目录下
- 可以通过修改脚本中的 `dataset` 变量切换不同数据集（human/celegans）

**生成的文件包括**：
- `compounds.npy`: 化合物原子特征
- `adjacencies.npy`: 化合物邻接矩阵
- `fingerprint.npy`: 化合物分子指纹
- `proteins.npy`: 蛋白质氨基酸序列特征
- `interactions.npy`: 相互作用标签
- `data_matrix.npy`: 化合物-蛋白质交互矩阵
- `compound_dict`, `protein_dict`: 索引映射字典
- `atom_dict`, `amino_dict`: 特征字典

### 2. 模型训练

使用默认参数训练 BiQBMVAE 模型：

```bash
python main.py -dataset human -use_qbm 1
```

使用传统 BiVAE 模型：

```bash
python main.py -dataset human -use_qbm 0
```

自定义超参数训练：

```bash
python main.py \
  -dataset human \
  -use_qbm 1 \
  -lr 0.0005 \
  -k 20 \
  -encoder_structure [40] \
  -likelihood pois \
  -gin_layers 3 \
  -batch_size 16 \
  -num_epochs 20
```

### BiQBMVAE vs BiVAE 实验对比

在 Human 数据集上的对比实验（使用相同的超参数）：

| 模型 | AUC | AUPR | F1 Score | 训练时间 |
|------|-----|------|----------|---------|
| BiVAE | 0.912 | 0.885 | 0.841 | 15分钟 |
| **BiQBMVAE** | **0.923** | **0.897** | **0.856** | 25分钟 |
| 性能提升 | +1.2% | +1.4% | +1.8% | +67% |
**结论**：
- BiQBMVAE 在所有指标上优于 BiVAE
- 训练时间增加约67%，但性能提升显著
- 对于追求最高精度的应用，BiQBMVAE 是更好的选择
!(https://github.com/Shaoruisun/kaiwu-pytorch-plugin/tree/main/experiments/batch_20251218_115445/analysis/training_curves.png)
## 模型架构

### 整体架构概览

BiVAE-CPI 采用**两阶段训练策略**：

**第一阶段：潜在表示学习**
- 使用 BiVAE 或 BiQBMVAE 从化合物-蛋白质交互矩阵中学习低维潜在表示
- 输出：每个化合物的潜在向量 θ 和每个蛋白质的潜在向量 β

**第二阶段：端到端CPI预测**
- 使用 BiBAECPI 模型结合潜在表示和分子/序列特征进行精确预测
- 输出：化合物-蛋白质对的相互作用概率

---

### BiVAE（双向变分自编码器）

BiVAE 是一个经典的协同过滤模型，通过对化合物-蛋白质交互矩阵进行双向分解，学习化合物和蛋白质的低维潜在表示。

#### 核心组件

1. **化合物编码器**（Compound Encoder）
   - 输入：交互矩阵的行向量（n_proteins 维）
   - 结构：多层感知机 (MLP)
   - 输出：均值 μ_θ 和标准差 σ_θ
   - 功能：将化合物的交互模式编码为 k 维潜在向量

2. **蛋白质编码器**（Protein Encoder）
   - 输入：交互矩阵的列向量（n_compounds 维）
   - 结构：多层感知机 (MLP)
   - 输出：均值 μ_β 和标准差 σ_β
   - 功能：将蛋白质的交互模式编码为 k 维潜在向量

3. **重参数化技巧**（Reparameterization Trick）
   ```
   θ = μ_θ + σ_θ ⊙ ε,  其中 ε ~ N(0, I)
   β = μ_β + σ_β ⊙ ε,  其中 ε ~ N(0, I)
   ```

4. **解码器**（Decoder）
   - 化合物重构：X̂ = sigmoid(θ × β^T)
   - 蛋白质重构：X̂^T = sigmoid(β × θ^T)

#### 损失函数

```
Loss = β_KL × KL(q(z|x) || p(z)) - E[log p(x|z)]
```

其中：
- **KL散度项**：确保潜在分布接近标准正态分布
- **重构项**：支持三种似然函数
  - `bern`：伯努利分布（二值交互）
  - `pois`：泊松分布（计数数据）
  - `gaus`：高斯分布（连续值）

---


#### BiQBMVAE 架构详解

##### 1. QBM编码器（QBMEncoder）

```python
输入: x ∈ R^n (去中心化的交互向量)
    ↓
全连接层: R^n → R^h (h为隐藏层维度)
    ↓
LayerNorm: 归一化隐藏层激活
    ↓
激活函数: ReLU/Tanh/Sigmoid
    ↓
全连接层: R^h → R^k (k为潜在维度)
    ↓
输出: z ∈ R^k (潜在表示)
```

**特点**：
- 使用 LayerNorm 而非 BatchNorm，适应小批量训练
- 可配置的激活函数，默认使用 ReLU
- L2权重衰减正则化，防止过拟合

##### 2. 量子玻尔兹曼机（RestrictedBoltzmannMachine）

这是 BiQBMVAE 的**核心量子组件**，来自Kaiwu 量子计算框架。

**数学原理**：
```
能量函数: E(v, h) = -∑ᵢ aᵢvᵢ - ∑ⱼ bⱼhⱼ - ∑ᵢⱼ vᵢWᵢⱼhⱼ

其中：
- v: 可见层节点（visible units）
- h: 隐藏层节点（hidden units）
- W: 权重矩阵
- a, b: 偏置项
```

**量子采样过程**：
1. **初始化**：将潜在向量映射到可见层
2. **Gibbs采样**：
   ```
   p(hⱼ=1|v) = sigmoid(bⱼ + ∑ᵢ vᵢWᵢⱼ)  # 正向传播
   p(vᵢ=1|h) = sigmoid(aᵢ + ∑ⱼ hⱼWᵢⱼ)  # 反向传播
   ```
3. **量子退火**：使用模拟退火优化器找到低能态
4. **输出**：优化后的潜在表示

**关键参数**：
- `num_visible=10`：可见层节点数
- `num_hidden=10`：隐藏层节点数
- `h_range=[-1, 1]`：磁场强度范围
- `j_range=[-1, 1]`：耦合强度范围

##### 3. 模拟退火优化器（SimulatedAnnealingOptimizer）

**退火调度**：
```
T(t) = T₀ × α^t

其中：
- T₀: 初始温度
- α = 0.95: 冷却系数
- t: 迭代步数
```

**优化过程**：
1. 高温阶段：广泛探索解空间（避免局部最优）
2. 降温阶段：逐渐收敛到全局最优
3. 接受准则：Metropolis准则
   ```
   P(接受新解) = exp(-ΔE/T)
   ```

##### 4. QBM解码器（QBMDecoder）

```python
输入: z ∈ R^k (潜在表示)
    ↓
全连接层: R^k → R^h
    ↓
LayerNorm: 归一化
    ↓
激活函数: ReLU/Tanh/Sigmoid
    ↓
全连接层: R^h → R^n
    ↓
输出: x̂ ∈ R^n (重构的交互向量)
```

#### BiQBMVAE 训练流程

##### 交替训练策略

```python
For epoch in range(epochs):
    # 阶段1: 训练蛋白质侧
    for batch_proteins in protein_batches:
        # 1. 编码
        β_encoded = QBM_Protein_Encoder(proteins)

        # 2. 量子优化
        β_optimized = RBM.quantum_sampling(β_encoded)
        β_optimized = Annealing.optimize(β_optimized)

        # 3. 解码
        proteins_reconstructed = QBM_Decoder(β_optimized, θ_fixed)

        # 4. 计算损失并更新
        loss = KL_divergence + Reconstruction_loss
        optimizer.step()

        # 5. 更新潜在表示
        β[protein_ids] = β_optimized

    # 阶段2: 训练化合物侧
    for batch_compounds in compound_batches:
        # 对称的过程
        θ_encoded = QBM_Compound_Encoder(compounds)
        θ_optimized = RBM.quantum_sampling(θ_encoded)
        compounds_reconstructed = QBM_Decoder(θ_optimized, β_fixed)
        ...
```

#### Kaiwu 框架集成

**导入模块**：
```python
from kaiwu.torch_plugin import RestrictedBoltzmannMachine, QVAE
from kaiwu.classical import SimulatedAnnealingOptimizer
```

**许可证配置**：
```python
import kaiwu as kw
kw.license.init("your_license_id", "your_secret_key")
```

**关键API**：
- `RestrictedBoltzmannMachine`：量子玻尔兹曼机实现
- `QVAE`：量子变分自编码器基类
- `SimulatedAnnealingOptimizer`：经典模拟退火算法（在经典硬件上模拟量子效应）

---

### BiBAECPI：端到端预测模型

BiBAECPI（Bilateral Autoencoder for Compound-Protein Interaction）是最终的预测模型，融合多源特征进行精确预测。

#### 架构组件

1. **化合物特征提取**

   **a) GIN图神经网络**
   ```
   输入: 分子图 G = (V, E)
       ↓
   Embedding: 原子类型 → 80维向量
       ↓
   GIN层1: 消息传递 + 聚合
       ↓
   GIN层2: 消息传递 + 聚合
       ↓
   GIN层3: 消息传递 + 聚合
       ↓
   图池化: 节点级 → 图级表示
       ↓
   输出: compounds_gnn ∈ R^k
   ```

   **b) Morgan分子指纹**
   ```
   输入: 1024位指纹 (FP2)
       ↓
   全连接: 1024 → 80
       ↓
   全连接: 80 → k
       ↓
   输出: compounds_fp ∈ R^k
   ```

2. **蛋白质特征提取**

   **CNN序列编码器**
   ```
   输入: 氨基酸序列 (n-gram编码)
       ↓
   Embedding: 氨基酸 → 80维向量
       ↓
   1D-CNN层1: kernel_size=11, channels=80
       ↓
   1D-CNN层2: kernel_size=11, channels=80
       ↓
   1D-CNN层3: kernel_size=11, channels=80
       ↓
   全局池化: 序列级 → 蛋白质级
       ↓
   输出: proteins_cnn ∈ R^k
   ```

3. **特征融合与预测**

   ```python
   # 获取BiQBMVAE学到的潜在表示
   θ_c = bivae.mu_theta[compound_id]  # 化合物潜在向量
   β_p = bivae.mu_beta[protein_id]    # 蛋白质潜在向量

   # 特征增强（逐元素乘法）
   feature_compound = θ_c ⊙ compounds_gnn  # 潜在表示引导的GNN特征
   feature_protein = β_p ⊙ proteins_cnn    # 潜在表示引导的CNN特征

   # 三路特征融合
   fusion = concat(feature_compound, feature_protein, compounds_fp)

   # 二分类预测
   output = Softmax(Linear(fusion))  # [P(无交互), P(有交互)]
   ```

## 超参数说明

### 基础参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-dataset` | human | 数据集名称（human/celegans） |
| `-mode` | gpu | 运行模式（gpu/cpu） |
| `-cuda` | 0 | GPU 设备 ID |
| `-use_qbm` | 1 | 是否使用量子增强模型（1: BiQBMVAE, 0: BiVAE） |

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-lr` | 0.0005 | 学习率 |
| `-batch_size` | 16 | 批次大小 |
| `-num_epochs` | 20 | 训练轮数 |
| `-step_size` | 10 | 学习率衰减步长 |
| `-gamma` | 0.5 | 学习率衰减率 |
| `-dropout` | 0.1 | Dropout 比例 |

### BiVAE / BiQBMVAE 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-k` | 20 | 潜在因子维度 |
| `-encoder_structure` | [40] | 编码器隐藏层结构 |
| `-likelihood` | pois | 似然函数（pois/bern/gaus） |
| `-act_fn` | relu | 激活函数 |

### QBM-VAE 专用参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-num_visible` | 10 | RBM 可见层节点数 |
| `-num_hidden` | 10 | RBM 隐藏层节点数 |
| `-dist_beta` | 10.0 | 分布参数 β |

### GIN 网络参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-gin_layers` | 3 | GIN 层数 |
| `-num_mlp_layers` | 3 | MLP 层数 |
| `-hidden_dim` | 50 | 隐藏层维度 |
| `-neighbor_pooling_type` | mean | 邻居聚合方式（sum/mean/max） |
| `-graph_pooling_type` | sum | 图级聚合方式（sum/mean/max） |

### 特征维度参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-comp_dim` | 80 | 化合物原子特征维度 |
| `-prot_dim` | 80 | 蛋白质氨基酸特征维度 |
| `-latent_dim` | 80 | 最终潜在表示维度 |

### CNN 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-window` | 5 | CNN 窗口大小 |
| `-layer_cnn` | 3 | CNN 层数 |

---

## 评估指标

模型在训练和测试过程中会输出以下评估指标：

- **AUC**（Area Under Curve）：ROC 曲线下面积，衡量分类性能
- **AUPR**（Area Under Precision-Recall Curve）：PR 曲线下面积
- **F1 Score**：精确率和召回率的调和平均
- **Precision**（精确率）：预测为正例中真正为正例的比例
- **Recall**（召回率）：真实正例中被正确预测的比例

训练过程中会在验证集上选择最佳模型（基于 AUC），并在测试集上报告最终性能。

## 技术支持

如有问题或建议，请通过以下方式联系：
- 邮件联系项目维护者：1047587695@qq.com
