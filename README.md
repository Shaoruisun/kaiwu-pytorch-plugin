# BiVAE-CPI：基于双向变分自编码器的化合物-蛋白质相互作用预测

## 项目简介

BiVAE-CPI 是一个基于深度学习的化合物-蛋白质相互作用（Compound-Protein Interaction, CPI）预测模型。该项目利用**双向变分自编码器（Bilateral Variational Autoencoder, BiVAE）**和**量子玻尔兹曼机变分自编码器（BiQBMVAE）**从化合物和蛋白质的交互矩阵中学习潜在表示，结合图神经网络和卷积神经网络对化合物和蛋白质进行特征提取，从而实现高精度的相互作用预测。

### 核心特性

- **双模型架构**：支持传统 BiVAE 和量子增强的 BiQBMVAE 两种模型
- **量子计算集成**：Kaiwu 量子计算框架，利用量子玻尔兹曼机增强特征学习
- **图神经网络**：使用 Graph Isomorphism Network (GIN) 处理化合物的分子图结构
- **端到端训练**：自动化的数据预处理、模型训练和评估流程
- **多数据集支持**：支持 Human 和 C.elegans 等多个生物数据集

---

## 环境要求

### 系统要求
- Python >= 3.9
- CUDA（可选，用于 GPU 加速）

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
kaiwu (Kaiwu 量子计算框架)
```
### 数据集
```
将该路径下https://github.com/YuBinLab-QUST/BiVAE-CPI/的data/ dataset/ 文件夹放入该路径即可
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


### 模型训练

使用默认参数训练 BiQBMVAE 模型：

```bash
python main.py -dataset human -lr 0.0005 -k 20 -use_qbm 1 -num_visible 10 -num_hidden 10
```
