---
title: Tracing Back to the Origins | Elman Network "Finding Structure in Time" (1990)
date: 2023-10-22 11:30:42
mathjax: true
tags:
  - Academics

---

Following the discussion of Cognitron (a neural network with feature extraction capabilities), we now turn to neural networks capable of processing language. This article explains one of the origins of recurrent neural networks (RNNs): the Elman Network proposed in Jeffrey Elman’s 1990 paper *Finding Structure in Time*. The recursive mechanism introduced by the Elman Network extended the capabilities of traditional feedforward neural networks, enabling the network to capture temporal dependencies in sequential data, thus laying a critical theoretical foundation for later models like Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU).

### Motivation: How to Improve the Representation of Time?

In past articles, we explored some neural networks designed for logical calculations and visual recognition (and the algorithms used to train these networks). However, a crucial factor in truly simulating intelligence and cognition remains — time.

Time is evidently significant in cognition; it is integral to many time-series behaviors, such as language and continuous-frame comprehension. In fact, it is challenging to imagine addressing basic issues like goal-oriented behavior, planning, or causality without some form of temporal dependency. However, introducing the concept of time in neural networks appears to conflict with the commonly used parallel processing models of the time, such as the Hopfield Network and Parallel Distributed Processing (PDP). In parallel processing paradigms, the computation of each hidden layer neuron depends only on the weighted sum of the input vector $\mathbf{x}$, without any sequential dependency among neurons, and is completed directly via matrix multiplication.

Nevertheless, parallel-processing neural networks are not entirely incapable of modeling sequential information; it just requires specific designs and mechanisms. We will continue discussing this in the next section.

### Issues with Parallel Processing in Sequence Modeling

One approach is to design the input as a fixed-length sequence, where each time step’s information corresponds to elements within the input vector. However, this spatialization of time has several significant problems:
1. It ignores the order and dynamic characteristics of the time series.
2. It lacks contextual memory.
3. It presents challenges in modeling time dependencies.

To solve these issues, Elman introduced a **context layer** $U_h$ based on the Perceptron network. The hidden layer state $\mathbf{h}(t)$ is determined by both the current input $\mathbf{x}(t)$ and the hidden state of the previous time step (i.e., the content in the context layer) $\mathbf{h}(t-1)$:

\[
\mathbf{h}(t) = \sigma \left( \mathbf{W}_h \mathbf{x}(t) + \mathbf{U}_h \mathbf{h}(t-1) + \mathbf{b}_h \right).
\]

The output $\mathbf{y}(t)$ at each time step is determined by the hidden state $\mathbf{h}(t)$:

\[
\mathbf{y}(t) = \sigma_o \left( \mathbf{W}_o \mathbf{h}(t) + \mathbf{b}_o \right).
\]

At the end of each time step, the current hidden state $\mathbf{h}(t)$ is copied into the context layer for the calculation in the next time step:

\[
\mathbf{c}(t) = \mathbf{h}(t)
\]

Like other neural networks, the Elman Network is trained using the Backpropagation algorithm. Below, we will showcase some of the network's capabilities as demonstrated in experiments.

### Experiments and Conclusions in the Paper

1. **Temporal XOR Problem**:
   - The XOR (exclusive OR) problem is a classic problem that cannot be solved by a simple two-layer feedforward neural network. When this problem is converted into a temporal format, the network must predict the XOR result of two consecutive bits.
   - **Motivation**: To explore whether recurrent neural networks (RNNs) can process time-based inputs and solve the temporal version of the XOR problem by using memory (context layer).
   - **Training Details**: The network receives a sequence of 3000 bits, with 1 input unit, 2 hidden units, 1 output unit, and 2 context units. The task is to predict the next bit based on the previous bits. The network was trained over 600 epochs on the entire sequence.
   - By using both the current input and the hidden layer's previous state (context), the network learned to predict the next bit. Unlike the static XOR problem, the network developed hidden units sensitive to repeated or alternating input patterns in the temporal version, solving it differently.

2. **Structure in Alphabet Sequences**:
   - A rule-based sequence of letters was generated, where consonants appear randomly, and each consonant is followed by a regular vowel pattern.
   - **Motivation**: To test whether the network could detect more complex temporal patterns and predict the next letter based on the structure of the input sequence.
   - **Training Details**: The input sequence consisted of 6-bit vectors representing six different letters. The network has 6 input units, 20 hidden units, 6 output units, and 20 context units. The task is to predict the next letter in the sequence, and the network was trained over 200 epochs.
   - The network successfully predicted the regular vowel patterns (which followed a fixed pattern) but struggled with the randomly distributed consonants. This indicates that the network could make partial predictions based on regularity in the input.

3. **Word Prediction Task (Discovery of the Concept of “Word”)**:
   - A continuous sequence of letters formed words and sentences without explicit word boundaries.
   - **Motivation**: To explore whether the network could implicitly learn the concept of “words” from the temporal structure of the input without explicit word boundary information.
   - **Training Details**: The input was a sequence of 4963 letters (generated from 200 sentences), with each letter represented by a 5-bit random vector. The network has 5 input units, 20 hidden units, 5 output units, and 20 context units, and the task is to predict the next letter.
   - The network’s error signals indicated boundaries between words, as the error was high at the beginning of new words and gradually decreased as the prediction became more predictable within a word. This suggests that the network could implicitly detect word structure based on co-occurrence statistics, even without explicit “word” instruction.

4. **Discovering Word Categories from Word Order**:
   - The order of words in sentences reflects grammatical constraints, but the network can only learn from the surface sequence of words. This experiment aims to investigate whether the network could learn syntactic and semantic categories of words solely from word order.
   - **Motivation**: To explore whether the network could learn abstract structural relationships (e.g., word classes, grammar) purely from word order.
   - **Training Details**: A sentence generator created 27,534 words (from 10,000 two- or three-word sentences). Each word was represented by a 31-bit random vector, and the network has 31 input units, 150 hidden units, 31 output units, and 150 context units. The task is to predict the next word.
   - The network developed internal representations that captured grammatical categories of words (e.g., nouns, verbs) and learned generalized patterns of word order. The hierarchical structure of hidden unit activations shows that the network could distinguish between different word types (e.g., animate vs. inanimate nouns) and represent abstract grammatical relationships.

### Conclusion

The experiments show that when problems are transformed into temporal forms, the nature of solutions changes. By developing memory mechanisms that store past inputs, the network successfully processes and predicts temporal data. The network uses error signals to guide learning, especially in cases with irregular or partially predictable temporal sequences. Additionally, the network developed hierarchical and context-sensitive representations, allowing it to capture general patterns among categories while also recognizing specific instances.