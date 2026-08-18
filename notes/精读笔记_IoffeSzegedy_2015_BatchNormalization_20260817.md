# 精读笔记：Batch Normalization (Ioffe & Szegedy, 2015)

## 一、论文速览

- **标题**：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- **作者**：Sergey Ioffe, Christian Szegedy (Google Inc.)
- **发表**：ICML 2015 (arXiv: 1502.03167)
- **定位**：深度学习训练方法论的范式级突破——首次形式化定义"内部协变量偏移"（ICS），并通过将归一化融入网络架构本身来解决，成为几乎所有现代深度网络的标配技术。
- **关键词**：Batch Normalization、Internal Covariate Shift、Mini-batch statistics、Whitening

---

## 二、核心概念

### 2.1 ICS（Internal Covariate Shift）

**定义**：训练过程中，由于网络参数变化导致的网络激活值分布的变化。

$$\text{参数更新} \;\rightarrow\; \text{下一层输入分布漂移} \;\rightarrow\; \text{ICS}$$

**ICS 的三个后果**：
1. 需要低学习率
2. 依赖谨慎初始化
3. 饱和非线性难训练（最致命——死锁循环）

**死锁循环**：

$$z \text{ 漂移到饱和区} \rightarrow g'(z) \to 0 \rightarrow \frac{\partial \mathcal{L}}{\partial W} \to 0 \rightarrow W \text{ 不更新} \rightarrow z \text{ 无法被拉回}$$

### 2.2 白化 vs 标准化

| | 标准化 | 白化 |
|---|--------|------|
| 零均值 | 是 | 是 |
| 单位方差 | 是 | 是 |
| 去相关 | 否 | 是（关键区别） |
| 计算代价 | 低 | 高（矩阵求逆） |

**标准化**（逐维度独立，BN 采用）：

$$\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{\mathrm{Var}[x^{(k)}]}}$$

**白化**（联合变换，论文理想方案）：

$$\hat{x} = \Sigma^{-1/2}(x - \mu)$$

**BN 的关键简化**：只做逐维度标准化、放弃去相关，用极低计算代价换取大部分收益。

### 2.3 白化不可行的原因

1. **计算不可行**：协方差矩阵 Σ 是 d×d 矩阵，矩阵求逆复杂度 O(d³)，每层每次前向传播都要算
2. **梯度不可处理**：白化的 μ 和 Σ 依赖 W 和 b，反向传播需对 Σ^{-1/2} 求导，无简单解析表达式
3. **白化反例**：归一化不在计算图里 → b 更新被抵消 → b 无限增长但 loss 不降

**一句话总结**：白化虽然是理想方案，但计算不可行且梯度无法正确处理，论文用它引出"归一化必须参与梯度计算"这一关键设计原则，然后放弃白化、改用更简单的 BN。

### 2.4 Hessian 条件数

**定义**：损失函数 ℓ 的 Hessian 矩阵 H 是二阶偏导矩阵，特征值为各主方向的曲率。条件数：

$$\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

- κ ≈ 1：各方向曲率均匀，SGD 稳定快速收敛
- κ ≫ 1：曲率极不均匀，学习率必须取到 λ_min 能容忍的小值

**与 ICS 的关系**：ICS 使各层输入分布不断漂移，导致不同层、不同参数方向的梯度量级差异巨大，Hessian 条件数恶化。BN 稳定分布 → 梯度量级均匀 → 条件数改善 → "improves the condition of the optimization problem"。

---

## 三、BN 算法（Algorithm 1）

### 3.1 设计目标

BN 对白化的两个修正：
1. 不在全训练集上做，改用 mini-batch 统计量
2. 归一化作为模型架构的一部分，让梯度流过它

### 3.2 四步公式

**输入**：mini-batch B = {x₁, ..., x_m}

**Step 1：计算均值**

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**Step 2：计算方差**

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

**Step 3：归一化**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

ε 是小常数（通常 1e-5），防止方差为零时除零。

**Step 4：可学习的缩放和平移**

$$y_i = \gamma \hat{x}_i + \beta$$

γ 和 β 是可学习参数，和 W、b 一样通过 SGD 更新。

**输出**：y_i = BN_{γ,β}(x_i)

### 3.3 γ, β 的设计哲学

**为什么需要 Step 4**：强制标准化到零均值、单位方差，把分布限制在固定范围内，限制了网络的表达能力。某些层可能需要非零均值的激活来更好地表达特征。

**γ, β 的作用**：
- γ 控制方差（分布的宽度）
- β 控制均值（分布的位置）
- 网络通过学习 γ 和 β，自己决定每层的最佳分布

**恒等变换性质**：若 γ = √(σ²_B + ε)，β = μ_B，则：

$$y = \gamma \hat{x} + \beta = \sqrt{\sigma_B^2 + \epsilon} \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \mu_B = x$$

BN 可表示恒等变换，因此**不降低网络容量**——加了 BN 层最坏情况和没加一样，不会更差，只会更好或一样。

> BN 不预设"归一化一定好"，而是让网络自己通过 γ, β 决定要不要归一化、归一化到什么程度。

---

## 四、训练与推理的区别

### 4.1 训练

- 用 mini-batch 统计量 μ_B, σ²_B（每次迭代都不同）
- mini-batch 统计量的随机波动注入噪声 → 正则化效果

### 4.2 推理

- 用训练全程的移动平均（固定值）：

$$\mu_{\text{test}} = E[\mu_B]$$

$$\sigma^2_{\text{test}} = \frac{m}{m-1} \cdot E[\sigma_B^2]$$

- 方差用 (m/(m-1)) 修正：因为 mini-batch 方差是有偏估计

### 4.3 为什么推理不能用 mini-batch 统计量

1. 单样本推理：线上服务通常一次只处理一张图，没有 mini-batch
2. batch 大小不一致：训练 batch=256，推理可能是 batch=1 或 batch=10
3. 推理要确定性输出，不要随机波动

### 4.4 "训练有噪声、推理无噪声"的通用模式

| 方法 | 训练时 | 推理时 |
|------|--------|--------|
| BN | μ_B 随机波动 | μ_test 固定 |
| Dropout | 随机置零 | 全部使用，权重缩放 |
| 数据增强 | 随机裁剪翻转 | 原图直接输入 |

逻辑：训练时制造困难（学鲁棒特征），推理时提供便利（稳定准确预测）。

---

## 五、卷积层的 BN 处理

### 5.1 设计原则

卷积层归一化要遵守卷积性质：同一特征图不同位置用同一归一化参数（权重共享 → 归一化也要共享）。

### 5.2 统计量计算

对每个特征图（通道），联合归一化 mini-batch 中所有位置：

- m 个样本 × p 个位置 = m×p 个元素
- γ, β 按特征图（通道）共享，每个通道一对

### 5.3 对比全连接层

| | 全连接层 | 卷积层 |
|---|---------|--------|
| 统计量计算范围 | mini-batch 的 m 个样本 | mini-batch 的 m 个样本 × p 个位置 |
| γ, β 数量 | 每个神经元一对 | 每个特征图（通道）一对 |

**数值示例**：batch=32，特征图 28×28，64 个通道
- γ, β 数量：64 对
- 统计量计算范围：32 × 784 = 25088 个元素

---

## 六、实验结果

### 6.1 MNIST 分布可视化

- 无 BN：激活分布持续漂移、形状变化（ICS 存在）
- 有 BN：分布稳定，接近初始分布（ICS 被解决）
- 意义：BN 的视觉验证——从实证上看到 ICS 被解决

### 6.2 BN-x5-Sigmoid 实验（最震撼）

- 无 BN 的 sigmoid 网络：几乎不训练，精度停在随机水平
- BN-x5 的 sigmoid 网络：成功训练，达到高精度
- 意义：问题不在 sigmoid 本身，而在 ICS。解决了 ICS，sigmoid 照样能用

### 6.3 ImageNet 结果

- 14 倍训练步数减少达到相同精度
- 显著超越原模型
- BN 同时起到正则化作用，消除 Dropout 需求

---

## 七、BN 的正则化机制

**关键纠正**：BN 的正则化不是来自降低参数量（BN 实际增加参数），而是来自 **mini-batch 统计量的随机性**。

训练时，一个样本的归一化结果取决于同一个 mini-batch 里的其他样本：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

每次迭代 mini-batch 组成不同 → μ_B 和 σ²_B 不同 → 同一个样本每次得到的归一化结果也不同 → 给隐藏激活注入噪声 → 抑制过拟合。

**如果用整个训练集**：μ 和 σ² 是固定值 → 没有随机性 → 没有正则化效果。

**加速与正则化不矛盾**：
- 加速来自稳定分布 → 改善条件数 → 大学习率
- 正则化来自 mini-batch 统计量的随机波动 → 注入噪声 → 抑制过拟合
- 一个管"优化好不好走"，一个管"泛化好不好"

---

## 八、BN 解决 sigmoid 训练问题的完整因果链

**第一步：问题根源**

参数更新 → 下一层输入分布漂移（ICS）→ z 扩散到饱和区 → sigmoid 梯度趋零 → 梯度消失 → 死锁

**第二步：BN 的介入点**

在 z = Wu + b 之后、激活函数 g(z) 之前插入 BN：

$$z = Wu + b \;\xrightarrow{\text{BN}}\; y = \gamma \hat{z} + \beta \;\xrightarrow{g}\; a = g(y)$$

BN 把 z 拉回零均值、单位方差，再由 γ, β 调整到最佳分布。

**第三步：死锁被打破**

$$z \text{ 被拉回非饱和区} \rightarrow g'(z) \approx 0.25 \rightarrow \text{梯度正常回传} \rightarrow W \text{ 正常更新} \rightarrow \text{训练继续}$$

**第四步：进一步加速**

分布稳定 → Hessian 条件数改善 → 可以用大学习率 → 5 倍学习率仍稳定训练

**一句话总结**：

> BN 不是换掉 sigmoid，而是从根源（分布漂移）解决问题——把 z 拉回非饱和区，让 sigmoid 的梯度恢复正常，死锁打破，深层网络重新可训练。

---

## 九、与其他方法的对比

### 9.1 初始化方法 vs BN

| | Xavier/He 初始化 | BN |
|---|-----------------|-----|
| 作用阶段 | 仅第 0 步 | 训练全程 |
| 机制 | 设定初始方差，让信号起点稳定 | 每步前向都重新归一化，维持分布稳定 |
| 性质何时成立 | 仅初始 | 每次前向传播 |
| 被训练破坏 | 是 | 否 |

**类比**：初始化是一次性的"起点修正"，BN 是每一步的"全程修正"——这也是 BN 标题里 "during training" 的含义。

### 9.2 正交初始化为什么难训练

1. **正交性一步就没了**：SGD 更新后 $(W - \eta\nabla)^T(W - \eta\nabla) \neq I$，第一次更新后正交性消失
2. **维持正交需要约束优化**：每步更新后都要投影回正交群，代价高
3. **卷积层不好构造**：卷积的等效矩阵是结构化稀疏的，构造严格正交变换麻烦
4. **正交保方差不保分布**：即使 W 全程正交，经过 ReLU 后均值照样偏移，ICS 依然存在

### 9.3 BN 的权重尺度不变性

$$\mathrm{BN}((aW)u) = \mathrm{BN}(Wu)$$

无论参数怎么被 SGD 更新、怎么缩放，每一层的输入分布在每一步都被重新拉回受控状态。

### 9.4 Adagrad vs BN

| | Adagrad | BN |
|---|---------|-----|
| 作用 | 按参数自适应学习率 | 稳定分布 |
| 层面 | 治标（补偿梯度量级差异） | 治本（消除分布漂移） |
| 局限 | 累积量单调增长，后期学习率趋零 | 无此问题 |

---

## 十、关键交互问答记录

### Q1：深度网络为什么比浅层难训练？

三点原因：
1. 线性变换改变分布：$x \sim \mathcal{N}(0,1) \Rightarrow z = Wx + b \sim \mathcal{N}(b, W^2)$
2. 非线性激活改变分布形状：sigmoid 输出 (0,1)，ReLU 截断负半轴使输出不再零均值
3. 参数级联变化：浅层 W₁、W₂ 持续更新 → 深层输入分布持续变

### Q2：sigmoid 的饱和区问题

- z=0 处导数 0.25（最大值）
- |z| > 2 后梯度很小
- 问题不是"sigmoid 在某点怎样"，而是训练中分布漂移到 |z| 很大的区域 → g'(z) → 0 → 梯度消失

### Q3：Jacobian 乘积

反向传播中，损失 ℓ 对第 l 层参数的梯度通过链式法则展开：

$$\frac{\partial \ell}{\partial W_l} = \frac{\partial \ell}{\partial z_L} \cdot \prod_{k=l}^{L-1} \frac{\partial z_{k+1}}{\partial z_k} \cdot \frac{\partial z_l}{\partial W_l}$$

每一项 $\frac{\partial z_{k+1}}{\partial z_k}$ 是第 k 层的 Jacobian 矩阵。连乘后，若大部分特征值 >1 则梯度爆炸，<1 则梯度消失。

### Q4：参数变化的逐层放大

$$\Delta z_L \approx \left(\prod_{k=1}^{L-1} W_{k+1} \cdot g'(z_k)\right) \cdot \Delta W_1 \cdot x$$

如果每层 |W · g'(z)| > 1，ΔW₁ 这个微小变化到达第 L 层时指数放大。

### Q5：prevent vs mitigate

| | 缓解（mitigate） | 预防（prevent） |
|---|-----------------|-----------------|
| ICS 是否存在 | 存在，只是补偿 | 不存在，被消除 |
| 代表方法 | 小学习率、谨慎初始化、Adagrad | BN |
| 作用机制 | 事后修正参数更新 | 事前稳定输入分布 |

---

## 十一、术语表

| 术语 | 英文全称 | 中文解释 |
|------|---------|---------|
| ICS | Internal Covariate Shift | 训练中各层激活分布因参数变化而发生的偏移 |
| Whitening | Whitening | 线性变换使数据零均值、单位方差、去相关 |
| γ, β | Scale and shift parameters | 可学习的缩放和偏移参数，使 BN 可表示恒等变换 |
| BN Transform | Batch Normalizing Transform | Algorithm 1 定义的完整归一化变换 |
| Saturating nonlinearity | Saturating nonlinearity | 饱和非线性：如 sigmoid/tanh，输入绝对值大时梯度趋零 |
| Hessian | Hessian matrix | 损失函数的二阶偏导矩阵，特征值描述各方向曲率 |
| Condition number | Condition number | κ = λ_max / λ_min，衡量优化问题的条件好坏 |

---

## 十二、精读完成日期

2026-08-17

至此五篇论文全部精读完成：Glorot 2010、He 2015、NIN 2014、GoogLeNet 2014、BN 2015。
