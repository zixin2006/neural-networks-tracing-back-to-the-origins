---
title: Tracing Back to the Origins | The Universal Approximation Theorem of Artificial Neural Networks
date: 2024-06-01 11:30:42
mathjax: true
tags:
  - Academics

---

![Image](images/nn.png)

In 1989, Cybenko published a paper titled *Approximation by Superpositions of a Sigmoidal Function*. In this paper, Cybenko proved that a feedforward neural network with a single hidden layer can approximate any continuous function defined on a compact set. This conclusion later became known as the **Universal Approximation Theorem**. In this paper, I will attempt to outline the entire proof process, clarify points of confusion encountered during reading, and delve into mathematical details Cybenko left unexplored.

**Problem Description**
A feedforward neural network with a single hidden layer typically has the following form:
$$
G(x) = \sum_{j=1}^{N} \alpha_j \sigma(w_j^T x + \theta_j)
$$
In the structure of a neural network, $w_j$ represents the weights applied to the input $x$, $\alpha_j$ represents the weights of the hidden layer outputs, and $\theta_j$ represents the bias for neuron $j$. We call $w_j^T x + \theta_j$ an **affine transformation** of the neural network input.

Let $I_n$ represent the $n$-dimensional unit cube $[0, 1]^n$, and let $C(I_n)$ denote the space of continuous functions on $I_n$, while $M(I_n)$ denotes the space of finite, regular Borel measures on $I_n$. We use $|f|$ to denote the supremum norm of $f \in C(I_n)$, with $|\cdot|$ commonly representing the maximum value of the function over its domain. The main objective of the proof is to explore the conditions under which functions of the form
$$
G(x) = \sum_{j=1}^{N} \alpha_j \sigma(w_j^T x + \theta_j)
$$
are dense in $C(I_n)$ with respect to the supremum norm.

**Definition 1** If for all $y \in \mathbb{R}^n$ and $\theta \in \mathbb{R}$, the integral over measure $\mu \in M(I_n)$
$$
\int_{I_n} \sigma(w^T x + \theta) d\mu(x) = 0
$$
implies $\mu = 0$, then $\sigma$ is **discriminatory**.

I found this definition somewhat confusing while reading. Another way to express it is: for a non-zero measure $\mu$, there exist $w_j, \theta_j$ such that
$$
\int_{I_n} \sigma(w^T x + \theta) d\mu(x) \neq 0
$$
If the above expression equals zero, then $\mu$ must be a zero measure. More intuitively, a discriminatory $\sigma$ is non-destructive in volume, ensuring that no information is lost when $x$ is processed after being weighted and shifted, and thus it does not reduce the affine space $w_j^T x + \theta_j$ to a zero-measure set. Next, let’s look at the definition of a **sigmoidal** function.

**Definition 2** A sigmoidal activation function satisfies
$$
\sigma(t) \to
\begin{cases}
1, & t \to +\infty \\
0, & t \to -\infty
\end{cases}
$$
There are many functions that meet this definition (such as logistic, softmax, etc.), but we will discuss the general case here.

**Main Conclusion**

**Theorem 1** Let $\sigma$ be any continuous discriminatory function. Then, finite sums of the form
$$
G(x) = \sum_{j=1}^{N} \alpha_j \sigma(w_j^T x + \theta_j) \tag{1}
$$
are dense in $C(I_n)$. In other words, for any $f \in C(I_n)$ and $\epsilon > 0$, there exists a $G(x)$ of the above form such that
$$
|G(x) - f(x)| < \epsilon, \quad x \in I_n.
$$
**Proof** Let $S \subset C(I_n)$ be the set of functions constructed by (1). Clearly, $S$ is a linear subspace of $C(I_n)$. To prove that $S$ is dense in $C(I_n)$, we need to show that the closure of $S$ is $C(I_n)$ itself. We denote the closure of $S$ by $\overline{S}$. Assuming $\overline{S} \neq C(I_n)$, then $\overline{S}$ is a proper closed subspace of $C(I_n)$.

**Hahn-Banach Theorem**  
Let $X$ be a real vector space, and $p$ be a sublinear bounded functional on $X$. Let $X_0$ be a linear subspace of $X$, and $f$ be a real linear functional on $X_0$. If $\forall x \in X_0$, $f(x) \leq p(x)$, then there exists a real linear functional $F$ on $X$ such that $F(x) \leq p(x)$ for all $x \in X$.

The Hahn-Banach theorem provides a way to extend a bounded linear functional defined on $S$ to a bounded linear functional defined on the entire $C(I_n)$. According to the Hahn-Banach theorem, there exists a bounded linear functional on $C(I_n)$, denoted by $L$, such that
$$
L(S) = 0, \quad L \neq 0.
$$

**Riesz Representation Theorem**  
Let $X$ be a locally compact Hausdorff space, and $C_c(X)$ be the set of continuous functions with compact support on $X$. For every positive linear functional $\Lambda$ on $C_c(X)$, there exists a unique positive and regular Borel measure $\mu$ such that for all $f \in C_c(X)$
$$
\Lambda(f) = \int_X f d\mu.
$$
The measure $\mu$ is regular and satisfies the following properties:
- For any open set $U \subset X$, $\mu(U) = \sup \{ I(f) : f \in C_c(X), f \in [0, 1], \text{supp} f \subset U \}$.
- For any compact set $K \subset X$, $\mu(K) = \inf \{ I(f) : f \in C_c(X), f \geq \chi_K \}$.

By the Riesz Representation Theorem, for some $\mu \in M(I_n)$ and any $h \in C(I_n)$, the functional $L$ can be expressed as follows:
$$
L(h) = \int_{I_n} h(x) d\mu(x).
$$
For any $w, \theta$, functions of the form $\sigma(w^T x + \theta)$ are in $S$, and $L$ is zero on $S$. Therefore,
$$
\int_{I_n} \sigma(w^T x + \theta) d\mu(x) = 0.
$$
Since $\sigma$ is a discriminatory function, we must have $\mu = 0$. This contradicts the conclusion $L \neq 0$, because
$$
\mu = 0 \implies \int_{I_n} h(x) d\mu(x) = 0, \quad h \in C(I_n).
$$
Thus, $S$ is dense in $C(I_n)$. Q.E.D.

**Lemma 1** Any continuous sigmoidal function is discriminatory.

To revisit the definition of a discriminatory function, we need to check the condition
$$
\int_{I_n} \sigma(w^T x + \theta) d\mu(x) = 0.
$$
This is an integral over a measure space, which can be simplified by constructing an increasing sequence of non-negative simple functions to approximate it.

Note that for any $x, w, \theta, \phi$, we have
$$
\sigma( \lambda(w^T x + \theta) + \phi) \to
\begin{cases}
1, & w^T x + \theta > 0 \text{ as } \lambda \to +\infty, \\
0, & w^T x + \theta < 0 \text{ as } \lambda \to +\infty, \\
\sigma(\phi), & w^T x + \theta = 0 \text{ for all } \lambda.
\end{cases}
$$
Thus, as $\lambda \to +\infty$, the function sequence $\sigma(\lambda(w^T x + \theta) + \phi)$ is bounded and converges pointwise to
$$
f(x) =
\begin{cases}
1, & w^T x + \theta > 0, \\
0, & w^T x + \theta < 0, \\
\sigma(\phi), & w^T x + \theta = 0.
\end{cases}
$$

Define the hyperplane $\Pi_{w,\theta} = \{ x \in \mathbb{R}^d : w^T x + \theta = 0 \}$ and the open half-space $H_{w,\theta} = \{ x \in \mathbb{R}^d : w^T x + \theta > 0 \}$. Note that for any $x$, we have $|\sigma_\lambda(x)| \leq \max(1, \sigma(\phi))$. Therefore, by the Lebesgue Dominated Convergence Theorem, we can obtain
$$
\int_{I_n} \sigma(x) d\mu(x) = \lim_{\lambda \to \infty} \int_{I_n} \sigma_\lambda(x) d\mu(x) = \int_{I_n} \gamma(x) d\mu(x)
$$
where
$$
\gamma(x) = \sigma(\phi) \mu(\Pi_{w,\theta}) + \mu(H_{w,\theta}).
$$
If the measure of every half-space is zero, then $\mu(\Pi_{w,\theta}) = 0, \mu(H_{w,\theta}) = 0$, and thus
$$
\int_{I_n} \sigma(w^T x + \theta) d\mu(x) = 0.
$$
Next, we prove that in this case, $\mu$ must be a zero measure.

Let $w$ be a fixed value. For a bounded measurable function $h$, define the linear functional $F : L^\infty \to \mathbb{R}$ as follows:
$$
F(h) = \int_{I_n} h(w^T x) d\mu(x).
$$
Since
$$
|F(h)| = \left| \int_{I_n} h(w^T x) d\mu(x) \right| \leq |h|_\infty \cdot \mu(K),
$$
and $\mu(I_n)$ is a finite measure, we conclude that $F$ is a bounded (continuous) functional.

Define $h$ as the indicator function on $[\theta, +\infty]$, that is
$$
h(x) =
\begin{cases}
1, & x \geq \theta, \\
0, & x < \theta.
\end{cases}
$$
Based on the assumption that the measure of any half-space is zero, we have
$$
F(h) = \int_{I_n} h(w^T x) d\mu(x) = \mu(\Pi_{y, -\theta}) + \mu(H_{y, -\theta}) = 0.
$$
Similarly, if $h$ is the indicator function on $(\theta, +\infty)$, $F(h) = \mu(H_{y, -\theta}) = 0$. If we let $h_E$ denote the indicator function on interval $E$, then we can write
$$
h_{[\theta_1, \theta_2]} = h_{[\theta_1, +\infty)} - h_{(\theta_2, +\infty)}, \quad h_{(\theta_1, \theta_2)} = h_{[\theta_1, +\infty)} - h_{[\theta_2, +\infty]}.
$$
Since $F$ is a linear functional, from the above expression, we conclude that $F(h_{[a, b]}) = F(h_{(a, b)})$ holds for any interval $[a, b]$ and $(a, b)$. If $f$ is a step function, $f = \sum_{n=1}^{N} a_n h_{E_n}$, then
$$
F(f) = \sum_{n=1}^{N} a_n F(h_{E_n}) = 0.
$$

Since step functions are dense in $L^\infty(\mathbb{R})$, for any $f \in C(I_n)$, there exists a sequence of step functions $r_n$ such that $r_n \to f$. By the continuity of $F$, we can deduce:
$$
F(f) = F\left(\lim_{n \to \infty} r_n \right) = \lim_{n \to \infty} F(r_n) = 0,
$$
which implies $F(\cdot) = 0$. 

In the above discussion, we assumed a constant value of $w$. Consider the Fourier transform $\hat{\mu}(w)$ of an arbitrary $w \in \mathbb{R}^n$:
$$
\begin{align*}
\hat{\mu}(w) &= \int_{I_n} e^{-iw^T x} d\mu(x) \\
&= \int_{I_n} \cos(w^T x) d\mu(x) - \int_{I_n} \sin(w^T x) d\mu(x) \\
&= F(\cos(\cdot)) - F(\sin(\cdot)) \\
&= 0.
\end{align*}
$$
If $\hat{\mu}(w) = 0$ holds for all $w \in \mathbb{R}$, then the measure $\mu$ must be zero. Therefore, any continuous sigmoidal function is discriminatory. Q.E.D.

Continuing with the main conclusions and implications for artificial neural networks:

**Main Conclusion and Artificial Neural Networks**

**Theorem 2** Let $\sigma$ be any continuous sigmoidal function. Finite sums of the following form
$$
G(x) = \sum_{j=1}^{N} \alpha_j \sigma(y_j^T x + \theta_j)
$$
are dense in $C(I_n)$. In other words, for any $f \in C(I_n)$ and $\epsilon > 0$, there exists a $G(x)$ of the above form such that
$$
|G(x) - f(x)| < \epsilon, \quad x \in I_n.
$$

**Proof** By combining Theorem 1 and Lemma 1, we observe that continuous sigmoidal functions satisfy the conditions of Theorem 2.

Thus, we have proven that a feedforward neural network with a single hidden layer can approximate any continuous function defined on a compact set. Function approximation plays a fundamental role in machine learning, especially in tasks like classification. Let $m$ be the Lebesgue measure on $I_n$, and let $(P_1, P_2, \dots, P_k)$ be disjoint measurable subsets of $I_n$. We can define a decision function $f$ as
$$
f(x) = j \quad \text{if and only if} \quad x \in P_j.
$$
If $f(x) = j$, then it must be that $x \in P_j$, thus completing the classification. The following theorem will demonstrate that such decision functions can be approximated by a single hidden-layer feedforward neural network.

**Theorem 3** Let $\sigma$ be a continuous sigmoidal function. Suppose $f$ is a decision function on any finite measurable subset of $I_n$. For any $\epsilon > 0$, there exists a function of the form
$$
G(x) = \sum_{j=1}^{N} \alpha_j \sigma(y_j^T x + \theta_j)
$$
and a subset $D \subset I_n$, such that $m(D) \geq 1 - \epsilon$, and for all $x \in D$, we have
$$
|G(x) - f(x)| < \epsilon.
$$

**Proof** According to **Lusin's Theorem**:

**Lusin's Theorem** If $E$ is a measurable subset of $\mathbb{R}^n$ and $g$ is a measurable function defined on $E$ and finite almost everywhere, then for any $\delta > 0$, there exists a closed set $F$ such that $m(E - F) < \delta$ and $g(x)$ is continuous everywhere on $E \cap F$.

Thus, there exists a continuous function $h$ and a subset $D$ with $m(D) \geq 1 - \epsilon$, such that for all $x \in D$, $h(x) = f(x)$. Since $h$ is continuous, by Theorem 2, we can find a function $G$ such that for all $x \in I_n$,
$$
|G(x) - h(x)| < \epsilon.
$$
Thus, for all $x \in D$, we have:
$$
|G(x) - f(x)| = |G(x) - h(x)| < \epsilon.
$$
Q.E.D.

**Conclusion**

After Cybenko’s paper, Hornik discovered in 1991 that the ability of neural networks to serve as universal approximators does not depend solely on the specific form of the activation function but rather on the diversity of neurons and structure of multilayer neural networks. Subsequent research has tested and validated the Universal Approximation Theorem for different activation functions and architectures. This theorem emphasizes that neural networks can approximate any complex function to arbitrary precision, though it does not prescribe a method for achieving this limit. Only by combining theory with practice can we approach the true potential of universal approximation.

**References**

[1] Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals, and Systems*, 2(4), 303–314. https://doi.org/10.1007/bf02551274