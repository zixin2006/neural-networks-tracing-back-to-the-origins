---
title: Tracing Back to the Origins | The MP Neuron "Logical Calculus of the Ideas Immanent in Nervous Activity" (1943)
date: 2023-05-29 11:30:42
mathjax: true
cover_image: images/tbto_11.png
tags:
  - Academics

---

# Introduction

This article discusses the paper *A Logical Calculus of the Ideas Immanent in Nervous Activity* [1], published in 1943 by biologist Warren McCulloch and logician Walter Pitts. In Chinese, it translates to *The Logical Calculus of Ideas in Nervous Activity*. Before reading this article, I suggest reading the introductory piece to better understand the symbolic systems used in the paper. Recently, I’ve been diving into some "classical" papers (often regarded as foundational in modern AI development). However, finding the content challenging and lacking detailed analysis, I managed to piece together a basic understanding during my spare time. I hope this “back-to-basics” article can aid learners with similar interests who may not know where to start. 😊

This article is around 5000 words, estimated reading time is 20 minutes.

If we were attending a general introduction to artificial intelligence, the instructor would likely start with artificial neural networks (ANNs) and introduce the basic structure of input layers, hidden layers, and output layers, as shown on the right side of Figure 1. The weighted sum of neuron outputs, passed through an activation function $f$, intuitively aligns with our understanding of biological structures and the basic mechanism of neuron pulse generation. But how did scientists abstract the complex circuitry of neural activity (Figure 1 left) into the simplified ANN structure? The 1943 paper *A Logical Calculus of the Ideas Immanent in Nervous Activity* is widely recognized as the earliest work addressing this question. This article's main contribution was the proposal of an idealized neuron model based on logical operations, allowing neurons to process information through threshold responses. It also demonstrated that neural networks could simulate any mechanical computing process and its invariance from a macro perspective. In short, as shown in Figure 1, it simplifies the complexity of neural physiological processes into logical propositions.

![Image](images/tbto_11.png)
<center>Figure 1: (Left) A high-resolution detailed image of a small part of the human brain mapped by Google in 2021 [2]; (Right) Basic structure of neural networks [3].</center>

## Introduction

The article begins with several key points in the Introduction (not analyzed here word-for-word, but you can search online for a PDF copy; the content, being rather old, can be difficult to read, especially for non-biology majors, like me):

1. Neural activity has an "all-or-none" nature (neural activity is either present or absent, akin to 0 or 1, true or false).
2. To trigger a pulse in the next neuron, a certain number of synapses must activate within a specific time frame (each neuron has a threshold independent of its position and previous activity).
3. The only significant delay in the nervous system is synaptic conduction delay (axon delay can be ignored).
4. Inhibitory synapses prevent neuron activation at certain times (explains the refractory period of neurons using inhibitory synapses).
5. The structure of neural networks does not change over time (long-term phenomena like learning and extinction are broadly equivalent to the original network).

Based on these assumptions, we define some symbols. If this part seems hard to understand, consider reading the preliminary article.

## Definitions

We first define a functor $S$ whose value for a given property $P$ is the property held when $P$ satisfies its predecessor. The expression is given by
$$
S(P)(t) = P(Kx), \quad t = x'
$$
where $K$ is a numerical operator, and $t$ denotes the successor of $x$. This expression may look complex, but it essentially means that $S(P)(t) = P(t-1)$. Here, the parameter in the parentheses on the left side is often omitted, making it a predicate, $[\textbf{Pr}]$, and $S^2 \mathbf{Pr} = S(S(\mathbf{Pr}))$.

Now, we define the structure of a neural network $\mathcal{N}$, which includes neurons $c_1, c_2, \ldots, c_n$. Let $N_i(t)$ represent the state of neuron $c_i$ at time $t$, describing whether neuron $c_i$ fires at time $t$. Sometimes, we use the object language and denote it as $N_i$. The boundary neurons of $\mathcal{N}$ are defined as neurons without axons connecting to their synapses. **Boundary neurons do not receive synaptic connections, so their input signals act directly on other parts of the network, unaffected by other neurons in the network ("inputs"). This property allows us to break down complex neural activities into simpler components for understanding**. Let $N_1, \ldots, N_p$ represent the boundary neurons and $N_{p+1}, \ldots, N_n$ represent the other neurons. The solution of such a neural network takes the form:
$$
S_i: N_{p+1}(z_1) \equiv Pr_i(N_1, \ldots, N_p, z_1)
$$
where $Pr_i$ contains only one free variable $z_1$ and may include some constant sentences $[\mathbf{sa}]$. Furthermore, each $S_i$ holds for $\mathcal{N}$. Next, we define two key terms: narrow realizability and extended realizability.

![Image](images/tbto_12.png)
<center>Figure 2: Original definition of narrow realizability (i.n.s.) and extended realizability [1].</center>

Oppositely, consider a predicate $Pr_1(^1{p_1}^1, ^1{p_2}^1, \ldots, ^1{p_p}^1, z_1, s)$. It is called narrowly realizable if there exists a network $\mathcal{N}$ with a series of $N_i$ such that $N_1(z_1) \equiv Pr_1(N_1, N_2, \ldots, z_1, sa_1)$ holds. Subsequently, we define extended realizability, meaning that $S^n(Pr_1)(p_1, \ldots, p_p, z_1, s)$ is narrowly realizable for some $n$.

To interpret these definitions, narrow realizability is a stringent condition requiring a specific neural network structure to directly achieve a particular state described by the predicate through specific neuron arrangements and input substitutions. Conversely, extended realizability is a more lenient condition, allowing multiple applications of functor $S$ to transform and expand propositions until the expanded proposition is narrowly realizable.

![Image](images/tbto_13.png)
<center>Figure 3: A simple neural network.</center>

Let’s consider a simple example (Figure 3). First, take the expression $N_3(z_1) \equiv Pr_(N_1, N_2, z_1)$. Since $N_3(t) \equiv N_1(t-1) \land N_2(t-1)$, this neural network is narrowly realizable by the i.n.s. definition. For $N_5(z_1) \equiv Pr(N_1, N_2, z)$, this expression is not directly realizable. Still, we can utilize intermediate neurons cleverly to derive a neural network since $N_5(t) \equiv N_4(t-1) \lor N_3(t-1)$, and according to $N_3(t) \equiv N_1(t-1) \land N_2(t-1)$, we get $N_5(t) \equiv N_1(t-2) \land N_2(t-2)$. As $S^2 Pr$ is narrowly realizable, $Pr$ is extendedly realizable, as shown in the neural network in Figure 3.

Finally, we arrive at the last definition! The paper defines Temporal Propositional Expressions (TPE) using recursive rules.

![Image](images/tbto_14.png)
<center>Figure 4: Recursive definition of Temporal Propositional Expressions (TPE) [1].</center>

First, $p[z_1]$ is the basic temporal propositional expression, where $p_1$ is a predicate variable and $z_1$ is a time point. Secondly, the definition states that the result of applying functor $S$, logical “or,” “and,” or “not” operations to TPEs $S_1$ and $S_2$ is still a TPE. Lastly, apart from the above forms, no other forms constitute TPEs—this ensures the completeness of the definition.

## Review

We defined functor $S$, solutions for neural networks, narrow and extended realizability, and TPEs. Before moving on to proofs and examples, let’s review the core questions:
1. Find an effective method to obtain a computable set of $S$ that forms the solution for any given network (calculate the behavior of any network).
2. Describe a set of realizable solutions.

In simple terms, the problems are (1) to calculate any network's behavior and (2) to determine networks that manifest as specific states.

## Theorems and Simple Proofs

*Note 1: A 0th-order neural network refers to a network without circular structures. The latter half of the paper provides detailed explanations for nets with circles, which are not covered here.*

**Theorem 1.** *Every 0th-order net can be solved in terms of Temporal Propositional Expressions (TPEs).*

Let $c_i$ be any neuron in $\mathcal{N}$ with a threshold $\beta_i > 0$. Let $c_{i1}, c_{i2}, \ldots, c_{ip}$ represent the neurons with excitatory synapses on $c_i$, with synaptic weights $n_{i1}, n_{i2}, \ldots, n_{ip}$. Let $c_{j1}, c_{j2}, \ldots, c_{jq}$ represent the neurons with inhibitory synapses on $c_i$. Let $\kappa_i$ be the class of subsets of $\{n_{i1}, n_{i2}, \ldots, n_{ip}\}$ such that the sum of these subsets exceeds $\beta_i$ (capable of activating $c_i$). Based on the assumptions introduced in the Introduction, we can write
$$
N_i(z_1) \equiv S \left\{\prod_{m=1}^q \lnot N_{jm}(z_1) \land \sum_{\alpha \in \kappa_i} \prod_{s \in \alpha} N_{is}(z_1) \right\}
$$
where $\sum$ and $\prod$ represent finite disjunction and conjunction, respectively. This expression may look complex, but its meaning is simple. It states that neuron $c_i$ fires at time $z_1$ if and only if at time $z_1 - 1$, none of the inhibitory neurons are firing and there exists a subset of excitatory neurons whose cumulative pulse value exceeds the threshold $\beta_i$, as illustrated in Figure 5.

![Image](images/tbto_15.png)
<center>Figure 5: A neural network composed of excitatory and inhibitory neurons.</center>

Assuming a threshold of two, it’s clear that $N_4$ firing requires both $N_1$ and $N_2$ to be true (i.e., $c_1$ and $c_2$ fired in the previous moment) and $N_3$ to be false (i.e., the inhibitory neuron $c_3$ was not activated). Since such an expression can be written for each non-boundary neuron, we can replace each $N_{jm}$ and $N_{is}$ with their equivalent expressions until $N_i(z_1)$ is entirely equivalent to a propositional logic expression formed by boundary neurons, thus deriving the solution of a neural network.

**Theorem 2.** *Every TPE is realizable by a 0th-order network in the extended sense.*

*Note 2: The term “realizable in the extended sense” is abbreviated as “realizable” in the proof.*

The second theorem states that for any neuron state description, there exists a 0th-order network that realizes it in the extended sense. The proof of Theorem 2 provides a recursive method for constructing a 0th-order network that realizes a TPE.

Since functor $S$ essentially acts as a time operator, it commutes with basic logic operations (disjunction, conjunction, negation). Thus, if network $\mathcal{N}$ can realize $S_i$ at the current time, we can achieve $S_i$ at the previous time by introducing appropriate delay neurons. Let’s look at an example:

Consider a proposition $S_i = p_1(z_1) \lor p_2(z_1)$, and suppose $S_i$ is narrowly realizable. Applying the functor $S$ and using its commutativity with logic operations, we get
$$
S(S_i) = S(p_1(z_1) \lor p_2(z_1)) = S(p_1(z_1)) \lor S(p_2(z_1))
$$
$S$ is a time operator, so $S(p_1(z_1))$ and $S(p_2(z_1))$ describe the states of $p_1(z_1)$ and $p_2(z_1)$ at the previous time point. By extending network $\mathcal{N}$ with appropriate delay neurons, we can realize $SS_i$ in the narrow sense.

![Image](images/tbto_16.png)
<center>Figure 6: Achieving a neural network by making neuron 3 an intermediate neuron.</center>

Thus, if $S_i$ is narrowly realizable, its result after $n$ applications of $S$ is still narrowly realizable because we can add intermediate neurons indefinitely. If $S_1$ and $S_2$ are both narrowly realizable, then $S^m S_1$ and $S^n S_2$ are also narrowly realizable. Now consider the four basic components of neural networks: $S(p_1(z_1))$, $S(p_1(z_1) \lor p_2(z_1))$, $S(p_1(z_1) \land p_2(z_1))$, and $S(p_1(z_1) \land \lnot p_2(z_1))$ (Figure 7), all of which are narrowly realizable.

![Image](images/tbto_17.png)
<center>Figure 7: Four basic components of a neural network.</center>

Based on the previous conclusions, $S^{m+n+1}(p_1(z_1))$, $S^{m+n+1}(p_1(z_1) \lor p_2(z_1))$, $S^{m+n+1}(p_1(z_1) \land p_2(z_1))$, and $S^{m+n+1}(p_1(z_1) \land \lnot p_2(z_1))$ are also narrowly realizable. By the definition of extended realizability, these primitive propositions are extendedly realizable. By logically combining these basic structures, more complex neural networks can be realized in the extended sense, enabling the realization of any TPE.

**Theorem 3.** *Let there be a complex sentence $S_1$ built up in any manner out of elementary sentences of the form $p(z_1-zz)$, where $zz$ is any numeral, by any of the propositional connections: negation, disjunction, conjunction, implication, and equivalence. Then $S_1$ is a* TPE *if and only if it is false when its constituent $p(z_1-zz)$ are all assumed false—i.e., replaced by false sentences—or that the last line in its truth table contains an 'F', or there is no term in its Hilbert disjunctive normal form composed exclusively of negated terms.*

Though complex at first glance, Theorem 3 provides the criterion for determining whether an expression is a TPE. In other words, it gives a method for determining if an expression is a TPE. There are three equivalent methods:
1. The composite sentence is false when all its basic sentences are false.
2. The last line of the truth table contains “false.”
3. There is no term in its Hilbert Disjunctive Normal Form (HDNF) composed solely of negated terms.

Let’s examine a few examples for each condition:
1. For the composite sentence $S_1 = p(z_1 - 1) \land q(z_1 - 2)$, where $p(z_1 - 1)$ and $q(z_1 - 2)$ are basic propositions. When both $p(z_1 - 1)$ and $q(z_1 - 2)$ are false, $S_1$ is also false, i.e., $S_1 = \text{False} \land \text{False} = \text{False}$. Therefore, $S_1$ satisfies the first condition and is a TPE.
2. For the composite sentence $S_2 = p(z_1 - 1) \lor q(z_1 - 2)$, the truth table is as follows:
   ![Image](images/tbto_18.png)
   As shown, when both $p(z_1 - 1)$ and $q(z_1 - 2)$ are false, $S_2$ is also false. Hence, $S_2$ satisfies the second condition.
3. For the composite sentence $S_3 = \neg p(z_1 - 1) \land \neg q(z_1 - 2)$, converting it to HDNF gives $S_3 = \neg p(z_1 - 1) \land \neg q(z_1 - 2)$. In HDNF, the only term $\neg p(z_1 - 1) \land \neg q(z_1 - 2)$ consists solely of negated terms. Therefore, $S_3$ does not meet the third condition and is not a TPE.

Now, let’s apply these theorems to implement a neural network. Consider a scenario: when an ice cube touches and then leaves our skin for a moment, we feel heat before coolness; but if it stays longer, we only feel a chill. To model this situation, let $N_1$ and $N_2$ represent “heat” and “cold” receptors, respectively, and let $N_3$ and $N_4$ represent neurons sensing heat and cold. We assume that $N_4$ senses cold only if the cold touch persists for two time units, yielding the following sentences:
$$
\begin{align*}
&N_3(t) \equiv N_1(t-1) \lor N_2(t-3) \land \lnot N_2(t-2) \\
&N_4(t) \equiv N_2(t-2) \land N_2(t-1)
\end{align*}
$$
Since these sentences are already in Hilbert Disjunctive Normal Form (HDNF), Theorem 3 indicates that both $N_3(t)$ and $N_4(t)$ are TPEs. Given this, Theorem 2 can construct a neural network in the following form:

![Image](images/tbto_19.png)

Finally, the paper mentions that different inhibition phenomena are broadly equivalent (2) extinction and learning are equivalent to absolute inhibition. If these two conclusions are proven, it indicates that the current structure simulates actual neural networks, allowing us to compute any network’s behavior and determine networks manifesting specific states. This part is briefly discussed in the original paper, so we will not explain the details it here.

## Summary

The McCulloch-Pitts neuron model simplifies neuron behavior into logical operations, executing basic logic through "all-or-none" responses. Its significance lies in formalizing the behavior of neural systems, demonstrating the Turing completeness of neural networks, capable of simulating any state. This groundbreaking theory laid a solid foundation for the Perceptron and propelled the development of artificial neural networks (ANNs). (Spoiler alert: the next topic will be Perceptron or theories related to emergence, expected within three days 🤞).

**End**

## References

[1] McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *The Bulletin of Mathematical Biophysics*, *5*(4), 115–133. https://doi.org/10.1007/bf02478259  
[2] Marshall, M. (2021, June 7). *Google has mapped a piece of the human brain in the most detail ever.* New Scientist; New Scientist. https://www.newscientist.com/article/2279937-google-has-mapped-a-piece-of-human-brain-in-the-most-detail-ever/  
[3] *A Friendly Introduction to [Deep] Neural Networks | KNIME*. (2021). KNIME. https://www.knime.com/blog/a-friendly-introduction-to-deep-neural-networks