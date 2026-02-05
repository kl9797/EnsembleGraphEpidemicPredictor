# EnsembleGraphEpidemicPredictor

EGEP: Epidemic Graph Ensemble Predictor
EGEP: 流行病图神经网络集成预测框架
EGEP is a robust library designed for spatiotemporal epidemic forecasting. It integrates state-of-the-art Spatiotemporal Graph Neural Networks (ST-GNNs) with a rigorous Forward Chaining Nested Cross-Validation stacking strategy.

EGEP 是一个专为时空流行病预测设计的稳健库。它集成了最先进的时空图神经网络 (ST-GNN)，并采用了严格的前向链式嵌套交叉验证堆叠 (Stacking) 策略。

## Key Features / 核心特性
### 1. Strict Forward Chaining Nested CV (严格的前向链式嵌套交叉验证)

Unlike traditional k-fold CV, EGEP respects the temporal order of data to prevent leakage. 不同于传统的 k-fold 交叉验证，EGEP 严格遵守数据的时间顺序以防止数据泄露。

Period A (Base Train): Train base learners (ST-GNNs)

Period B (Meta Train / Validation): Base learners predict unseen data. These predictions train the Meta Learner. 

Period C (Meta Test): Final evaluation on strictly future data.

### 2. Diverse Base Learners (多样化的基模型)

Built-in support for multiple PyTorch Geometric Temporal models: 内置支持多种 PyTorch Geometric Temporal 模型：

LSTM-based: lrgcn, gclstm, dygrencoder, gconvlstm

GRU/GCN-based: dcrnn, evolvegcno, gconvgru

Adaptive: agcrn (Adaptive Graph Convolutional Recurrent Network)

### 3. Adaptive Stacking (自适应stacking)

Dynamic Tuning: Uses Optuna to tune XGBoost hyperparameters for every temporal fold.

Lag Features: Automatically generates lag features and handles negative value correction.

动态调优： 使用 Optuna 为每一个时间切片 (Fold) 动态寻找 XGBoost 的最优超参数。

滞后特征： 自动构建滞后特征 (Lag Features) 并处理负值修正。

## Usage / 使用指南
### 1. Prepare Data (准备数据)

Data should be a list of PyTorch Geometric Data or StaticGraphTemporalSignal snapshots. 数据应为 PyTorch Geometric 的 Data 快照列表。

### 2. Run the Stacking Pipeline (运行堆叠流程)

```python
from egep.engine import StackingOrchestrator
from egep.utils.metrics import calculate_metrics
#1. Initialize Orchestrator / 初始化编排器
orchestrator = StackingOrchestrator(
    base_model_names=['lrgcn', 'dcrnn', 'agcrn'],
    node_features=35,
    num_nodes=212,
    device='cuda',  # or 'cpu'
    hidden_dim=32
)

#2. Run Nested CV / 运行嵌套交叉验证
#This strictly follows: Train(A) -> Meta Train(B) -> Test(C)
#严格遵循：训练(A) -> 元学习训练(B) -> 测试(C)
preds, actuals = orchestrator.run_nested_cross_validation(
    data_list=my_data_list,
    initial_train_window=28,  # Period A length
    meta_train_window=14,     # Period B length
    meta_test_window=14,      # Period C length
    stride=7,                 # Sliding step
    base_epochs=100,
    meta_trials=20            # Optuna trials
)

#3. Evaluate / 评估
metrics = calculate_metrics(preds, actuals)
print(metrics)
```

## Workflow Diagram / 流程图示
The core logic of the StackingOrchestrator is defined as follows: StackingOrchestrator 的核心逻辑如下所示：

```text
Time Axis:  [0 ........................................ T]

Split 1:
[Period A (Base Train)] -> [Period B (Meta Train)] -> [Period C (Meta Test)]
       (Train GNNs)      (GNN Preds -> Train XGB)   (Final Eval)

Split 2 (Slide Forward):
       [   Period A (Base Train)   ] -> [Period B] -> [Period C]
Phase 1: Train Base Models on Period A.

Phase 2: Use Base Models to predict Period B. Use these predictions (features) + True Values of B (targets) to tune & train the Meta Learner.

Phase 3: Use Base Models to predict Period C. Feed these into the trained Meta Learner to get final predictions.
```
## Directory Structure / 目录结构
```text
egep/
├── __init__.py
├── engine.py           # Core Orchestrator (Orchestrates the Split A/B/C logic)
├── callbacks.py        # Early Stopping utilities
├── models/
│   ├── __init__.py
│   ├── registry.py     # Model registration decorator
│   └── recurrent_gcn.py # Implementation of 8 base learners (LRGCN, AGCRN, etc.)
├── meta/
│   ├── __init__.py
│   └── meta_learner.py # XGBoost + Optuna + Lag Feature Generation
└── utils/
    ├── __init__.py
    └── metrics.py      # RMSE, MAPE, RAE calculation

```
