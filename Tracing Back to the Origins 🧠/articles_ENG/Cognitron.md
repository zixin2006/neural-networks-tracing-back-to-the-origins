---
title: Tracing Back to the Origins | Cognitron "A Self-organizing Multilayered Neural Network" (1975)
date: 2023-12-01 11:30:42
mathjax: true
tags:
  - Academics

---

In the last article on the perceptron, we discussed how the backpropagation algorithm adjusts the states of hidden units, allowing neural connections to learn and adapt their states, thus solving linearly inseparable problems. Now, we introduce a neural network with visual capabilities (such as feature extraction) and explore the early concepts that hint at Convolutional Neural Networks (CNNs). This article will cover the 1975 paper by Japanese computer scientist Kunihiko Fukushima titled *Cognitron: A Self-organizing Multilayered Neural Network* [1]. 

**Quick Summary**

The key innovations of this paper are as follows:
- Introduced a new hypothesis for synaptic organization in neurons: **Only if a neuron $y$ is activated by neuron $x$ and no other nearby neuron has a stronger response, will the synapse between $x$ and $y$ strengthen**.
- Introduced the concept of **receptive fields**, using two-dimensional neural network layers where each neuron receives input from a specific area of the previous layer. Based on this new hypothesis, it compares activation strength and strengthens synaptic connections, layer by layer, to build complex feature representations.
- Based on these hypotheses, the author derived a new algorithm to effectively organize multilayer neural networks, creating a self-organizing network called Cognitron. Cognitron’s advantage lies in its ability to **learn in an unsupervised manner** by automatically adjusting synaptic weights, enabling higher-level feature extraction from local features and achieving self-organized recognition capabilities for complex patterns.

The following sections will discuss the author's motivations, modeling approach, and results in detail.

**Introduction**

The paper opens with an explanation of **neural plasticity**: the synaptic connections of neurons in the brain are not entirely genetically determined but are malleable and shaped by learning or postnatal experiences. Studies by Hubel and Wiesel demonstrated that neurons in the visual cortex of normal adult cats exhibit selective sensitivity to lines and edges in the visual field, with preferences uniformly distributed across all directions. However, kittens raised in an environment consisting only of black and white stripes did not develop neurons responsive to directions perpendicular to those stripes (Blake & Cooper, 1970). This indicates that a lack of certain visual experiences can result in impaired neuronal responses, suggesting that **the response characteristics of visual cortex neurons are adjusted through visual experiences during development**. In neural networks, this natural plasticity is equivalent to **self-organization**, meaning the network can update weights and adjust synaptic connections without supervision.

At this point, you might recall the neuron properties in the perceptron, which seem to allow for self-adjustment of synaptic connections. However, because the original perceptron only has four parts—the input layer, projection area, association area, and response layer—and only the last two parts are randomly modifiable, the entire neural network does not have complete self-organizing capabilities. The perceptron was highly anticipated at the time but proved less powerful than initially hoped.

![Image](images/tbto_41.png)
**Figure 1: Rosenblatt's 1957 Perceptron [2].**

What about adding more layers? As we know, adding more layers to a neural network can significantly enhance its ability to extract higher-level information. However, when Fukushima wrote this paper, there was no algorithm to enable self-organization in multilayer neural networks (although this was later achieved by the backpropagation algorithm). Therefore, systems claiming self-organization did not fundamentally go beyond the three-layer perceptron framework.

To solve these problems and achieve unsupervised learning, the author proposed a new hypothesis to model synaptic strengthening.

**Hypothesis**

Before introducing the new hypothesis, let’s examine the issues with prior assumptions, which the author categorized into three types:

- Synapse $c_i$ starts in a modifiable state, but if postsynaptic neuron $y$ is activated without the activation of presynaptic neuron $x_i$, $c_i$ becomes inactive and no longer modifiable.
- Synapse $c_i$ has a random initial component, and only when both presynaptic cell $x$ and postsynaptic cell $y$ are activated, does $c_i$ strengthen. This type of synapse is known as a Brindley synapse.
- The postsynaptic cell $y$ has another synaptic input $z$, known as a Hebbian synapse, which is initially inactive and only strengthens when both presynaptic cell $x$ and the control signal $z$ are active (this model assumes $z$ as a "supervisor").

![Image](images/tbto_42.png)
**Figure 2: Mechanisms of three synapse types.**

The first assumption is highly problematic: if a single incorrect signal occurs, the synapse could irreversibly change, potentially rendering the network functionally "extinct" over time. Brindley synapses offer partial self-organization, but the randomness of initial synapses may not ensure meaningful connection patterns, and large-scale networks have not proven Brindley synapse effectiveness. Hebbian synapses rely on an external supervisor signal $z$, which is biologically unreasonable.

To address these issues, the author proposed a new hypothesis. For synapse strengthening between neurons $x$ and $y$ via synapse $c_i$, two conditions are required:

1. Presynaptic neuron $x$ of synapse $c_i$ is activated.
2. None of the neurons near postsynaptic neuron $y$ respond more strongly than $y$.

Condition two implies that synaptic strengthening is unique within its local neighborhood. This uniqueness allows each neuron in the network to develop a distinct response, enhancing the network’s **feature differentiation ability**. Moreover, if a neuron malfunctions, other neurons can take over its role, similar to self-repair functions in biological neural networks. This new hypothesis, akin to the brain’s mechanism of supplying nutrients only to the most responsive neurons, also has biological plausibility.

Next, let’s model the neurons and network structure.

**Neuron Modeling**

The neurons in the Cognitron use a "shunting inhibition" mechanism. In traditional linear inhibition, if an excitatory signal $E$ and an inhibitory signal $I$ are present, the final pulse strength is generally represented as $E - I$. Shunting inhibition refines this response, preventing over-activation by using a division-based model. The final output behaves similarly to an activation function, producing non-negative values proportional to the pulse strength. Denote $u(1), \ldots, u(N)$ as inputs from excitatory synapses and $v(1), \ldots, v(M)$ as inputs from inhibitory synapses. Each neuron’s output $w$ is defined as:

$$w = \varphi \left[ \frac{1 + \sum_{v=1}^{N} a(v) \cdot u(v)}{1 + \sum_{μ=1}^{M} b(μ) \cdot v(μ)} - 1 \right]$$

where $a(v)$ and $b(μ)$ are the conductance values for excitatory and inhibitory synapses, both non-negative. $\varphi[x]$ is defined as:
$$
\varphi[x] = \begin{cases}
x & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}
$$
Let $e$ represent the sum of all excitatory effects and $h$ the sum of all inhibitory effects:
$$
e = \sum_{v=1}^{N} a(v) \cdot u(v), \quad h = \sum_{μ=1}^{M} b(μ) \cdot v(μ)
$$
The neuron output formula can be simplified as:
$$
w = \varphi \left[ \frac{1 + e}{1 + h} - 1 \right] = \varphi \left( \frac{e - h}{1 + h} \right)
$$
In the Cognitron, synaptic conductance values $a(v)$ and $b(μ)$ increase over the learning process (which is intuitive); as conductance increases and $e \gg 1$ and $h \gg 1$, the above formula approximates:
$$
w = \varphi \left( \frac{e/h - 1}{1/h} \right)
$$
At this point, the output depends on the ratio $e/h$, not on their difference. Thus, even if conductance increases with learning, as long as excitatory synapses $a(v)$ and inhibitory synapses $b(μ)$ increase proportionally, the neuron output converges to a stable value instead of diverging. We assume excitatory and inhibitory inputs increase proportionally, represented by:
$$
e = \epsilon x, \quad h = \eta x
$$
where $x$ is the total signal strength. If $\epsilon > \eta$, the neuron output can be transformed as:
$$
\begin{align*}
w = \frac{(\epsilon - \eta)x}{1 + \eta x} &= \frac{\epsilon - \eta}{2\eta}\cdot \frac{2\eta x}{1+\eta x}\\
&=\frac{\epsilon - \eta}{2\eta}\cdot \left[1+\frac{e^{\ln \eta x}-1}{e^{\ln \eta x}+1}\right ]\\
&=\frac{\epsilon - \eta}{2\eta} \left[1 + \tanh \left(\frac{1}{2} \ln \eta x \right)\right]
\end{align*}
$$
We find that this input-output relationship is consistent with the Weber-Fechner law’s logarithmic relationship, expressed as an S-shaped response curve represented by the tanh function.

![Image](images/tbto_43.png)
**Figure 3: The Weber-Fechner Law – perceptual curve vs. physical reality.**

This formula is often used as an empirical formula in neurophysiology to approximate the nonlinear input-output relationships in sensory organs and animal sensory systems. The author believes that since these types of neural elements closely resemble biological neurons, they should be well-suited to various visual and auditory processing systems.

**Cognitron Structure**

Based on the new hypothesis, let’s dive into the structure of Cognitron. Cognitron is composed of multiple neural layers with similar structures, arranged sequentially. The $l$-th layer (labeled $U_l$) consists of excitatory neurons $u_l(\mathbf{n})$ and inhibitory neurons $v_l(\mathbf{n})$, where $\mathbf{n} = (n_x, n_y)$ represents the two-dimensional position of neurons.

The excitatory neuron $u_l(\mathbf{n})$ receives pulses from excitatory neurons $u_{l-1}(\mathbf{n+v}) [\mathbf{v}\in S_l]$ in layer $U_{l-1}$ and from inhibitory neurons $v_{l-1}(\mathbf{n})$. Here, $S_l$ represents the connection area for the neuron, and $\mathbf{n+v}$ represents the coordinates of all excitatory neurons in $S_l$. If we denote the synaptic conductance as $a(\mathbf{v, n})$ and $b(\mathbf{n})$, the output of $u_l(\mathbf{n})$ can be given by the following formula:
$$
u_l(\mathbf{n}) = \varphi \left[ \frac{1 + \sum_{v \in S_l} a(\mathbf{v, n}) \cdot u_{l-1}(\mathbf{n+v})}{1 + b(\mathbf{n}) \cdot v_{l-1}(\mathbf{n})} - 1 \right]
$$
The inhibitory neuron $v_{l-1}(\mathbf{n})$ receives weighted input from neighboring excitatory neurons $u_{l-1}(\mathbf{n+v})$ and outputs to $u_l(\mathbf{n})$:
$$
v_{l-1}(\mathbf{n}) = \sum_{\mathbf{v} \in S_l} c_{l-1}(\mathbf{v}) \cdot u_{l-1}(\mathbf{n+v})
$$
where $c_{l-1}(\mathbf{v})$ represents the weight of the inhibitory synapse, with the sum of weights equal to 1:
$$
\sum_{v \in S_l} c_{l-1}(v) = 1
$$
Figure 4 shows the connections between $U_{l-1}$ and $U_l$.

![Image](images/tbto_44.png)
**Figure 4: Visualization of Cognitron Structure.**

It can be seen that the receptive field of neuron $u_l(\mathbf{n})$ overlaps with the receptive fields of the excitatory synapses connected to $u_{l-1}(\mathbf{n})$. Now, let’s model the synaptic strengthening mechanism. Based on the hypothesis, let $\delta(\mathbf{n})$ be a Boolean function indicating whether or not a synapse should be strengthened. If $u_l(\mathbf{n})$ responds more strongly than any other neuron within neighborhood $\Omega_l$, it takes a value of 1:
$$
\delta(\mathbf{n}) = \begin{cases}
1 & \text{if} \ \forall \ \mathbf{v} \in \Omega_l, \ u_l(\mathbf{n}) \geq u_l(\mathbf{n+v}) \ \\
0 & \text{otherwise}
\end{cases}
$$
When $\delta(\mathbf{n}) = 1$, the changes in $\Delta a(\mathbf{v, n})$ and $\Delta b(\mathbf{n})$ depend on whether $u_l(\mathbf{n})=0$ or $u_l(\mathbf{n}) > 0$.

1. When $u_l(\mathbf{n}) = 0$
$$
\begin{align*}
\Delta a(\mathbf{v, n}) &= q_0 \cdot c_{l-1}(\mathbf{v}) \cdot u_{l-1}(\mathbf{n+v}) \cdot \delta(\mathbf{n}) \\
\Delta b(\mathbf{n}) &= q_0 \cdot v_{l-1}(\mathbf{n}) \cdot \delta(\mathbf{n})
\end{align*}
$$
2. When $u_l(\mathbf{n}) > 0$
$$
\begin{align*}
\Delta a(\mathbf{v, n}) &= q_1 \cdot c_{l-1}(\mathbf{v}) \cdot u_{l-1}(\mathbf{n+v}) \cdot \delta(\mathbf{n}) \\
\Delta b(\mathbf{n}) &= \frac{\sum_{\mathbf{v} \in S_l} q_1 \cdot c_{l-1}(\mathbf{v}) \cdot u_{l-1}^2(\mathbf{n+v})}{2v_{l-1}(\mathbf{n})} \cdot \delta(\mathbf{n})
\end{align*}
$$
$q_0$ and $q_1$ are positive constants, with $q_1 > q_0$. In the first case, if $u_l(\mathbf{n})=0$ and no neurons in the neighborhood respond, the synaptic strengthening amount is relatively small. In the second case, when $u_l(\mathbf{n}) > 0$, since $q_1 > q_0$, the synaptic strengthening is more significant, with $\Delta b(\mathbf{n})$ suppressed by the **square** of the input signal from the previous layer, thereby moderating the inhibitory synaptic strengthening amount, in line with the "winner-takes-all" rule in the hypothesis.

**Quantitative Analysis of the Algorithm**

The author conducted extensive analysis based on the Cognitron algorithm; here, we summarize some key insights without delving into the detailed analysis:

1. After learning, the number of neurons with strong responses significantly decreased, exhibiting **sparsity**, which helps the network distinguish between different input patterns.
2. When $u_l(\mathbf{n}) > 0$, excitatory synapses tend to strengthen more than inhibitory synapses; when $u_l(\mathbf{n}) = 0$, inhibitory synapses may strengthen more.
3. Repeated exposure to the same stimulus enhances the output connections between neurons $u_l(\mathbf{n})$. As $a(\mathbf{v,n})$ gradually increases, the output $w$ approaches 1, indicating that the network has "learned" a specific output pattern.

In addition to the basic algorithm, the author modeled lateral inhibition phenomena, but for simplicity, this is not covered here. Lateral inhibition reduces the response of surrounding neurons when a neuron responds strongly to a specific stimulus, promoting sparse connections.

**Layer Connections and Axonal Branching**

Finally, let’s define how the network layers connect. This paper discusses three different methods for defining the receptive field of each layer. The first method maintains the same receptive field size at each layer, but to achieve sufficient receptive field coverage, more layers are required, complicating the network structure. The second method increases the receptive field size layer by layer, allowing for larger receptive fields with fewer layers, but the similarity of responses in the final layer (output layer) weakens the network’s ability to distinguish different stimuli.

![Image](images/tbto_45.png)
**Figure 5: Three methods of defining layer connections.**

The third method (5c), chosen in this paper, uses probabilistically distributed axonal branches as the layers deepen, extending the receptive field without excessive overlap.

Cognitron adopts the third method. Suppose each excitatory neuron $u_l(\mathbf{n})$ has $K+1$ axonal branches, where one branch transmits directly forward, while other branches undergo probabilistic displacement. Let $P_{lk}$ be the permutation operator for $\mathbf{n}$. We have:
$$
u^{\prime}_l(\mathbf{n}, k) = P_{lk} \{ u_l(\mathbf{n}) \} \quad (k \neq 0)
$$
In computer simulations, this study focused on $K = 1$, where the axon divides into two branches, one direct and the other probabilistic.

The output of axonal branched neurons is redefined as:
$$
u^{\prime}_l(\mathbf{n}) = \varphi \left[ \frac{1 + \sum_{k=0}^{K} \sum_{v \in S_l} a(\mathbf{v, n}, k) \cdot u^{\prime}_{l-1}(\mathbf{n+v}, k)}{1 + b_l(\mathbf{n}) \cdot v_{l-1}(\mathbf{n})} - 1 \right]
$$
Other formulas largely remain the same, with the following substitutions:
$$
\begin{align*}
u_{l-1}(\mathbf{n+v}) &\rightarrow u^{\prime}_{l-1}(\mathbf{n+v}, k) \\
a(\mathbf{v, n}) &\rightarrow a(\mathbf{v, n}, k) \\
c_{l-1}(\mathbf{v}) &\rightarrow c_{l-1}(\mathbf{v}, k) \\
\sum_{\mathbf{v} \in S_l} &\rightarrow \sum_{\mathbf{v} \in S_l} \sum_{k=0}^{K}
\end{align*}
$$

**Computer Simulation and Conclusion**

In the simulation, the author used four network layers, each with $12\times12$ excitatory neurons and the same number of inhibitory neurons. The connection area $S$, neighborhood area $\Omega$, and range of lateral inhibition were all carefully set. The network was exposed to 12x12 images of numbers 0 through 4, with responses recorded at each layer.

![Image](images/tbto_46.png)
**Figure 6: Response patterns for numbers 0-4.**

The study found that after multiple exposures, Cognitron could achieve self-organization, with most cells in the deepest layer ($U_3$) selectively responding to specific stimulus patterns.

![Image](images/tbto_47.png)
**Figure 7: (Left) Reverse reconstruction from the normal responses of a single layer of cells. The first row shows the normal response to the stimulus “4,” recorded during the 20th cycle of pattern presentation. The second, third, and fourth rows respectively show the results of reverse reconstruction from the normal response patterns in layers U1, U2, and U3. (Right) Reverse reconstruction from the normal response of a single cell. The first row shows the normal response to the stimulus “4.” The second, third, and fourth rows respectively show the reverse reconstruction results from the strongest responses of individual cells in layers U1, U2, and U3.**

To verify the effect of synapse organization, researchers conducted a reverse reconstruction experiment, where the flow of information through synapses was assumed to be reversed, allowing them to observe the responses of each layer’s cells. Results showed that using this method, it was possible not only to **deduce specific numbers from the responses of 144 neurons but also to backtrace input patterns from single cells in deeper layers to earlier layers**, demonstrating that each neuron’s response was unique. This indicates that Cognitron developed a self-organizing ability specific to this task. (Results shown in Figure 7)

![Image](images/tbto_48.png)
**Figure 8: Cognitron’s network layer and single neuron reverse inference capabilities when receiving similar stimulus patterns.**

In the final set of experiments, as shown in Figure 8, the author explored Cognitron’s ability to respond to similar stimulus patterns (e.g., “X,” “Y,” “T,” and “Z”) using the same testing method as described above. Results showed that even though these letters shared common components (e.g., the head of X and Y, or the tail of Y and T), Cognitron was still able to distinguish between and respond differently to them, indicating its capacity to differentiate similar information.

**A Short Summary**

To summarize, by following a new synaptic organization hypothesis ("winner takes all, loser gets none!" albeit with a hint of humor), Cognitron successfully achieved self-organized learning, with many aspects of its algorithm displaying similarities to the biological brain. Due to its multi-layered structure, Cognitron could handle complex tasks in information processing more effectively than traditional brain models or previous artificial neural networks. However, research also pointed out that Cognitron lacks a complete capability for pattern recognition. For example, to fully enable Cognitron to perform pattern recognition tasks, additional functions such as spatial pattern normalization or completion would be needed—somewhat akin to our aim to achieve AGI. Yet, as we see even in 2024, the road ahead remains challenging.

Following Cognitron, Kunihiko Fukushima authored another paper introducing "Neocognitron," an upgraded version of Cognitron designed to make neural network recognition invariant to factors such as image rotation. Feel free to read it if you’re interested!

**Conclusion**

The "Tracing Back" series is now drawing to a close. In the next issue, we’ll cover the foundational language model paper *A Neural Probabilistic Language Model*, followed by a final issue with code reproductions of key papers. While writing these articles, I noticed an interesting trend: the earlier the work, the more it emphasized the connection between the “biological brain” and the “artificial brain.” In the Cognitron paper, we can see traces of neuroscientific influence throughout, which the researchers leveraged. The previous article on backpropagation, while closer to modern AI, already showed fewer of these connections. Nevertheless, even in that paper, the authors mentioned that since the biological brain does not follow backpropagation, we should look for more "natural" learning algorithms. 

Today, most AI research is categorized as a purely computational field, aiming for novel engineering ideas and constantly pushing SOTA in various projects. However, there’s a notable lack of research focused on understanding "why it works." This shift sometimes raises questions about our goals: has the purpose of our research diverged from the past? Are we still on the path to achieving AGI? We’ll leave a question mark here and let time reveal the answer.

Lastly, creating these articles has not been easy, so if you’ve read this far, please give it a like! (*＾3＾*)

**References**

[1] Fukushima, K. (1975). Cognitron: A self-organizing multilayered neural network. _Biological Cybernetics_, _20_(3–4), 121–136. https://doi.org/10.1007/bf00342633

[2] Singh, K., Ahuja, A., Chatterjee, T., Pritam, S., Varma, N., Jain, R., Sachdeva, M. S., Bansal, D., Trehan, A., Hajra, S., Kar, R., Basu, D., Peepa, K., Anil, I., Banashree, R., Apabi, H., Ghosh, S., Samanta, S., Chattopadhay, A., & Bhattacharyya, M. (2019). 60th Annual Conference of Indian Society of Hematology & Blood Transfusion (ISHBT) October 2019. _Indian Journal of Hematology and Blood Transfusion_, _35_(S1), 1–151