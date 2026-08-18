# 精读笔记：MobileNet v1 (Howard et al., 2017)

## 一、论文速览

- **标题**：MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- **作者**：Andrew G. Howard, Menglong Zhu, Bo Chen, et al. (Google)
- **发表**：arXiv 2017
- **定位**：移动端高效 CNN 的奠基之作——将标准卷积拆为深度可分离卷积，计算量降至原来的 1/8~1/9，开启移动端深度学习时代。
- **关键词**：Depthwise Separable Convolution、Width Multiplier、Resolution Multiplier、Mobile

---

## 二、痛点：标准卷积太贵

标准卷积计算量随通道数平方增长，移动/嵌入式设备算力有限，ResNet/DenseNet 等网络无法直接部署。

---

## 三、核心 Idea：深度可分离卷积

### 3.1 标准卷积在干两件事

标准卷积核 $D_K \times D_K \times M \times N$ **同时做**：

1. **空间滤波**：$D_K \times D_K$ 核捕捉局部空间模式
2. **通道融合**：把 $M$ 个输入通道线性组合成 $N$ 个输出通道

### 3.2 拆成两步

**第一步：Depthwise（深度卷积）——只做空间滤波**

每个输入通道单独配一个 $1 \times D_K \times D_K$ 核，通道间不混合。$M \to M$ 通道。

**第二步：Pointwise（逐点卷积）——只做通道融合**

用 $N$ 个 $M \times 1 \times 1$ 核（1×1 卷积），$M \to N$ 通道。

| | 核形状 | 通道变化 | 职责 |
|---|---|---|---|
| depthwise | $1 \times D_K \times D_K$，共 $M$ 个 | $M \to M$ | 空间滤波 |
| pointwise | $M \times 1 \times 1$，共 $N$ 个 | $M \to N$ | 通道融合 |

---

## 四、计算量推导（数学重点）

### 4.1 标准卷积

$$\text{标准} = D_K^2 \cdot M \cdot N \cdot D_F^2$$

### 4.2 深度可分离

$$\text{DW} = D_K^2 \cdot M \cdot D_F^2$$

$$\text{PW} = M \cdot N \cdot D_F^2$$

$$\text{DWS} = D_F^2 \cdot M\,(D_K^2 + N)$$

### 4.3 降比

$$\frac{\text{DWS}}{\text{标准}} = \frac{1}{N} + \frac{1}{D_K^2}$$

$D_K = 3$，$N \gg 9$ 时：

$$\frac{\text{DWS}}{\text{标准}} \approx \frac{1}{9}$$

### 4.4 数学本质：乘积变求和

**标准**：空间成本 $\times$ 通道成本（耦合）：

$$\underbrace{D_K^2}_{\text{空间}} \times \underbrace{M \cdot N}_{\text{通道}} \times D_F^2$$

**深度可分离**：空间成本 $+$ 通道成本（解耦）：

$$\underbrace{D_K^2 \cdot M}_{\text{depthwise}} + \underbrace{M \cdot N}_{\text{pointwise}} = M\,(D_K^2 + N)$$

> 标准卷积让"空间"和"通道"互相买单（相乘），深度可分离让它们各付各的（相加）。乘号变加号是计算量骤降的根源。

---

## 五、两个部署旋钮

### 5.1 Width Multiplier $\alpha$

$\alpha \in (0,1]$，$M \to \alpha M$，$N \to \alpha N$：

$$\text{DWS}_\alpha \approx \alpha^2 \cdot \text{DWS}$$

### 5.2 Resolution Multiplier $\rho$

$\rho \in (0,1]$，$D_F \to \rho D_F$：

$$\text{DWS}_\rho = \rho^2 \cdot \text{DWS}$$

两者叠加：$\alpha^2 \cdot \rho^2$ 倍计算量。

---

## 六、为什么拆开不丢精度？

**解耦假设**：空间相关性（局部纹理）和跨通道相关性（特征组合）是可以分开学的两种信息。

**为什么 $1/D_K^2$ 的冗余可以省**：标准卷积里 $N$ 个输出通道各自配一整套空间核，但空间滤波模式在不同通道间高度重叠。depthwise 让每通道只配一个空间核，不重复 $N$ 次。

---

## 七、实验结果

| 模型 | 参数量 | 乘加数 | ImageNet top-1 |
|---|---|---|---|
| MobileNet (α=1.0, 224) | 4.2M | 569M | 70.6% |
| MobileNet (α=0.75, 224) | 2.6M | 209M | 68.4% |
| GoogLeNet | 6.8M | 1300M | 70.6% |
| VGG-16 | 138M | 15300M | 71.5% |

MobileNet 用 65% 的参数、44% 的计算量达到 GoogLeNet 同等精度。

---

## 八、遗产

- 深度可分离卷积成为移动端 CNN 标准积木
- MobileNet v2（倒残差 + 线性瓶颈）、v3（NAS + SE）建立其上
- 解耦原则影响 ShuffleNet、Xception、EfficientNet
- 宽度/分辨率乘子成为标准部署旋钮

---

## 九、与其他方法的对比

### 9.1 MobileNet vs 标准卷积网络

| | 标准卷积 | 深度可分离 |
|---|---|---|
| 空间滤波 | ✓ | ✓（depthwise） |
| 通道融合 | ✓ | ✓（pointwise 1×1） |
| 两者关系 | 耦合（相乘） | 解耦（相加） |
| 计算量 | $D_K^2 MN D_F^2$ | $M(D_K^2+N)D_F^2$ |
| 降比 | 1 | $\approx 1/9$ |

### 9.2 与 NIN 的联系

NIN 首次用 1×1 卷积做跨通道融合，MobileNet 把"空间卷积 + 1×1 卷积"从优化技巧提升为整体架构原则——**所有卷积层都用深度可分离替换**。

### 9.3 Bottleneck 套路的变体

| 论文 | 1×1 卷积的角色 |
|---|---|
| NIN | 跨通道融合（mlpconv） |
| GoogLeNet/ResNet/DenseNet | 降维 bottleneck（辅助 3×3） |
| MobileNet | **独立承担通道融合**（不再是辅助） |

MobileNet 让 1×1 卷积从"降维辅助"升格为"正式的一步"。

---

## 十、关键交互问答记录

### Q1：M×N 这个平方项的物理意义是什么？

每个输出通道（共 $N$ 个）都需要看所有 $M$ 个输入通道，所以是 $M \times N$——输入输出通道间的"全连接"。标准卷积让空间核 $D_K^2$ 和这个 $M \times N$ 相乘，意味着空间滤波被重复 $N$ 次 -> 有几个卷积核就重复几次空间滤波。

### Q2：depthwise separable 怎么"降次"？

把 $D_K^2 \cdot M \cdot N$（乘积）拆成 $D_K^2 \cdot M + M \cdot N$（求和）。空间滤波只做 $M$ 次（每通道一次），通道组合只做 1×1（不乘 $D_K^2$）。乘号变加号。

---

## 十一、术语表

| 术语 | 英文 | 解释 |
|---|---|---|
| Depthwise Conv | Depthwise Convolution | 每通道独立空间卷积，不混合通道 |
| Pointwise Conv | Pointwise Convolution | 1×1 卷积，做通道线性组合 |
| Depthwise Separable | Depthwise Separable Conv | depthwise + pointwise 的组合 |
| Width Multiplier | $\alpha$ | 压缩通道数，计算量 $\sim \alpha^2$ |
| Resolution Multiplier | $\rho$ | 压缩特征图尺寸，计算量 $\sim \rho^2$ |

---

## 十二、精读完成日期

2026-08-17
