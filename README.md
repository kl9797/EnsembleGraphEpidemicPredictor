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

### 3. Adaptive Meta-Learning (自适应元学习)

Dynamic Tuning: Uses Optuna to tune XGBoost hyperparameters for every temporal fold.

Lag Features: Automatically generates lag features and handles negative value correction.

动态调优： 使用 Optuna 为每一个时间切片 (Fold) 动态寻找 XGBoost 的最优超参数。

滞后特征： 自动构建滞后特征 (Lag Features) 并处理负值修正。
