---
title: 2025-01-08-Generative-Models
date: 2025-01-08 09:29:17
tags:
---
<!-- www
## 01-08
### Histogram 
### Kernel Method
Kernel Density Estimation (KDE) 是一种统计方法，用于估计一个随机变量的**连续的**概率密度函数（Probability Density Function, PDF）。

直方图有个问题：

- **它是离散的**：你只能看到固定的时间间隔，比如10:00-11:00，11:00-12:00。
- **它依赖分箱（bin）大小**：不同的分箱方式会影响直方图的形状。

**KDE 就是一种连续的分布估计方法**，它不依赖于分箱，而是用一个 **“平滑的曲线”** 来表示数据的分布。  
你可以把 KDE 想象成：

- 每个数据点都画一个 **小山丘（kernel）**。
- 然后把这些小山丘 **叠加起来**，得到一个连续的曲线，代表数据的分布。

KDE 的公式为：
\(\hat{f}(x) = \frac{1}{n h} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)\)

**解释每个符号**：

- **\( \hat{f}(x) \)**：估计的概率密度函数。
- **\( n \)**：样本数量。
- **\( h \)**：平滑参数，称为 **带宽（bandwidth）**。它决定了小山丘的“宽度”。
- **\( K \)**：**核函数（Kernel Function）**，决定了小山丘的形状。

核函数是 KDE 的核心。常见的核函数有：

1. **高斯核（Gaussian Kernel）**  
   形状像正态分布的小山丘，最常用的核函数。

   \[
   K(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
   \]

2. **箱形核（Box Kernel）**  
   每个数据点生成一个“方形的小山丘”，边界清晰，但不平滑。

3. **三角核（Triangle Kernel）**  
   生成一个三角形的小山丘。

带宽（Bandwidth）对 KDE 的影响**

**带宽（h）** 是 KDE 中一个非常重要的参数。它决定了每个小山丘的 **宽度**，从而影响最终曲线的平滑程度。

- **带宽小**：曲线更贴近数据，但容易出现“过拟合”。
- **带宽大**：曲线更平滑，但可能会忽略一些细节。

### Series Methods
Imagine you're trying to approximate a complex function \( f(x) \) by expressing it as a combination of simpler functions (called **basis functions**). This is like saying:

\[ f(x) = \sum_{j=1}^{\infty} \beta_j \phi_j(x) \]

Where:
- \( \phi_j(x) \) are the **basis functions** (like sine, cosine, polynomials, etc.)
- \( \beta_j \) are the **coefficients** that tell us how much of each basis function to use.

---

### 🧩 **Orthogonal Basis**  
The basis functions \( \phi_j \) are said to be **orthogonal** if they satisfy the condition:

\[ \int \phi_j(x) \phi_k(x) dx = 0 \quad \text{for} \quad j \neq k \]

Think of orthogonal basis functions like directions that are at right angles to each other — they don't overlap in their effects.

---

### 📚 **Cosine Basis Example**
The notes give an example of using **cosine functions** as the basis:

\[ \phi_j(x) = \sqrt{2} \cos(2\pi j x) \]

These functions oscillate and can capture periodic patterns in the data.

---

### 📈 **Wavelet Basis**
There's also a reference to **wavelet basis functions**, which are used to capture both **local** and **global** structures in the data.

---

### 🔍 **Key Formula for Density Estimation**  
The board shows that if we are estimating a **density function** \( f \), we can approximate it using the formula:

\[ \hat{f}(x) = \sum_{j=1}^{k} \hat{\beta}_j \phi_j(x) \]

Where:
- \( \hat{\beta}_j = \frac{1}{n} \sum_{i=1}^{n} \phi_j(X_i) \)  
  (This is the estimated coefficient based on sample data \( X_i \).)

---

### 🧠 **Key Idea: Selecting the Right \( k \)**
Choosing the right number of basis functions (\( k \)) is important. If \( k \) is too large, the model will **overfit** (too complicated). If \( k \) is too small, the model will **underfit** (too simple).




## 01-15


## Part 1: Risk Analysis

### Background and Setup

You have \( N \) points \( x_1, x_2, \ldots, x_N \in \mathbb{R}^d \). This means you have \( N \) data samples, each one is a \( d \)-dimensional vector. We assume these points lie in some space \( X \subset \mathbb{R}^d \), which is said to be a **compact set** (meaning it is closed and bounded in the mathematical sense).

### Key Smoothness Assumption

You mentioned an assumption:

\[
D^s p(x) - D^s p(y) \;\le\; L \,\|x - y\|
\]

1. **\( D^s p(x) \)**:  
   - This notation represents a high-order derivative (partial derivative) of a probability density function \(p(x)\).  
   - The superscript \( s \) is a **multi-index**.  
     - For instance, \( s = (s_1, s_2, \dots, s_d) \).  
     - Then \( D^s p(x)\) means we take the \((s_1 + s_2 + \cdots + s_d)\)-th partial derivative of \( p \) with respect to each coordinate appropriately:
       \[
       D^s p(x) \;=\; \frac{\partial^{\,s_1 + s_2 + \cdots + s_d} p(x)}{\partial x_1^{s_1}\,\partial x_2^{s_2}\,\cdots\,\partial x_d^{s_d}}.
       \]

2. **Interpretation of the inequality**:  
   \[
   D^s p(x) - D^s p(y) \;\le\; L\, \|x - y\|.
   \]  
   This says that **the change in the \(s\)-th derivative** between \(x\) and \(y\) is bounded by their distance times some constant \(L\). It suggests that \(p(x)\) is **smooth**—no wild jumps in its high-order derivatives. This is often used in probability density estimation and ensures continuity and differentiability in a controlled way.

3. **Why Compactness Matters**:  
   - If \( X \) is compact, it’s easier to control or bound various integrals and derivatives since \( x \) cannot go off to infinity.  
   - This is important in risk analysis because it helps with bounding error terms and ensuring integrals converge.

> **Reference for multi-index notation**:  
> - [Brilliant.org: Multi-Index Notation](https://brilliant.org/wiki/multi-index-notation/)  
> - [Wikipedia: Multi-index](https://en.wikipedia.org/wiki/Multi-index)  

---

## Part 2: More on \( D^s p(x) \) and Multi-Index Notation

If \( p(x) \) is a function of multiple variables, say \( p(x_1, x_2, \dots, x_d) \), then  
\[
s = (s_1, s_2, \dots, s_d)
\]  
tells us **how many times** we differentiate with respect to each variable \( x_i \).  

For example, if \( s = (2, 1, 0, \dots, 0) \), then  
\[
D^s p(x) \;=\; \frac{\partial^3 p(x)}{\partial x_1^2 \,\partial x_2^1}.
\]

### Intuitive View
- If \(s_1 = 2\), that means “take the second derivative with respect to \(x_1\)”.  
- If \(s_2 = 1\), that means “take the first derivative with respect to \(x_2\)”.  
- We multiply these partial derivatives together in the correct order.

This notation is just a concise way to keep track of all the derivatives in multiple dimensions.

---

## Part 3: Taylor Expansion (Taylor’s Theorem)

A general **Taylor expansion** around a point \( x \) says that if \(f\) is sufficiently smooth, we can write:

\[
f(x + h) 
\;=\;
f(x) \;+\; 
\sum_{k=1}^{m} \frac{1}{k!} \bigl( D^k f(x) \bigr) (h,\dots,h) \;+\; R_m,
\]

where \( R_m \) is the remainder term, and \( D^k f(x) \) is the \(k\)-th derivative of \(f\). In multiple dimensions, we typically see it in terms of partial derivatives:

\[
f(x + h) \;\approx\; 
f(x) 
\;+\; \nabla f(x) \cdot h
\;+\; \frac{1}{2} h^\top \nabla^2 f(x) \, h
\;+\; \dots
\]

This expansion is used **a lot** in analyzing the bias of estimators, numerical methods, and even in bounding differences between function values.

> **Reference for Taylor expansions**:  
> - [Wikipedia: Taylor's theorem](https://en.wikipedia.org/wiki/Taylor%27s_theorem)  
> - [MathWorld: Taylor Series](https://mathworld.wolfram.com/TaylorSeries.html)  

---

## Part 4: Kernel Density Estimation (KDE)

You also mentioned **Kernel Density Estimation**. It’s a method to estimate an unknown probability density function \( f(x) \) from data. The estimator looks like:

\[
\hat{f}(x) \;=\; \frac{1}{N h} \;\sum_{i=1}^{N} K\Bigl(\frac{x - X_i}{h}\Bigr),
\]

where
- \(K\) is a **kernel function** (often something like a Gaussian, Epanechnikov, etc.),  
- \(h\) is the **bandwidth** (a smoothing parameter).

### 4.1 Kernel Function Properties

A kernel \(K\) often satisfies:

1. \(\int K(u)\,du = 1\).  
2. \(\int u\,K(u)\,du = 0\). (Centered around 0)  
3. \(\int u^r K(u)\,du = 0\) for odd \(r\) if it’s symmetric, etc.

An example of a kernel is the **Epanechnikov kernel**:

\[
G(x) \;=\; \frac{3}{4}\,\bigl(1 - x^2\bigr) \, 1(\lvert x \rvert \le 1),
\]

where \(1(\lvert x \rvert \le 1)\) is the indicator function that is 1 if \(\lvert x \rvert \le 1\) and 0 otherwise. 

> **Reference for Kernel Density Estimation**:  
> - [“Kernel Density Estimation Explained” on Towards Data Science](https://towardsdatascience.com/kernel-density-estimation-explained-52045f84c726)  
> - [A Comprehensive Guide to KDE on Analytics Vidhya](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-kernel-density-estimation/)  

---

### 4.2 Bias of the KDE

The **bias** of an estimator \(\hat{f}(x)\) is:

\[
\text{Bias}(\hat{f}(x)) \;=\; 
E[\hat{f}(x)] \;-\; f(x).
\]

For KDE, we often find that:

\[
E[\hat{f}(x)]
\;=\; 
\int K\!\Bigl(\frac{x - t}{h}\Bigr) p(t)\, \frac{dt}{h}.
\]

By doing a change of variable and performing a Taylor expansion of \(p(\cdot)\), we eventually get:

\[
E[\hat{f}(x)]
\;=\; 
p(x) 
\;+\; 
\frac{h^2}{2}\,\mu_2(K)\,p''(x) 
\;+\; 
O(h^4),
\]

where \(\mu_2(K)\) is the second moment of the kernel. So the **leading term** of the bias is:

\[
\text{Bias}(\hat{f}(x))
\;=\;
\frac{h^2}{2}\,\mu_2(K)\,p''(x)
\;+\;
O(h^4).
\]

This tells us that the bias grows (roughly) like \(h^2\). If \(h\) is too big, the bias is large (over-smoothing), but if \(h\) is too small, you get high variance (under-smoothing). That’s why **bandwidth selection** is super important.

> **Reference for bias-variance in KDE**:  
> - [Lecture notes on Kernel Density Estimation (CMU)](https://www.stat.cmu.edu/~cshalizi/402/lectures/08-kernel-density/kde.pdf)  


## 02-28
Cold Diffusion


## 03-10
突然觉得自己悟了？？
How others put them together
https://www.youtube.com/watch?v=B-d_3xX6ss4
根据song yang的论文的时间线。
1; sliced score model (定义了score based model)
(https://proceedings.mlr.press/v115/song20a/song20a.pdf)

2; annealed Langevin Dynamics
(https://papers.neurips.cc/paper_files/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html)
langevin mcmc sample; https://zhuanlan.zhihu.com/p/797467112
the whole thing; https://yang-song.net/blog/2021/score/

3; Score-based GM through SED (Song et al. ICLR 2021)
(https://arxiv.org/pdf/2011.13456)
SED 视角统一； 
smld (ve) 和 ddpm (vp)
sed和of-ode有一些对应关系

4; EDM 
（https://proceedings.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf）
pf-ode视角统一了diffusion 模型

5; CM
用了edm

6; 







 -->
