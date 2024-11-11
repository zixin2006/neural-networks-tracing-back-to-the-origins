---
title: Predicates, Functors, and Logical Connectives
date: 2023-04-25 11:30:42
mathjax: true
tags:
  - Academics

---

###### Predicates

**Predicates** are tools in logic for describing the attributes and relations of objects. A unary predicate describes the property of a single object, containing only one variable, typically represented in the form $P(x)$. For example, let $P(x)$ represent "x is red," so if $x$ is an apple, $P(x)$ indicates that this apple is red.

Binary and multiary predicates describe relationships between objects. For instance, $R(x, y)$ can represent $x > y$, so $R(5, 3)$ indicates that 5 is greater than 3. Similarly, $Pr(N_1, N_2, \ldots, N_p, z)$ might denote the activity state of neurons $c_1, c_2, \ldots, c_p$ at time $z$.

###### Functors

**Functors** are tools for establishing connections between different categories. In category theory, a category consists of objects and morphisms; objects can be anything, while morphisms are mappings that preserve structure between objects. Imagine two different cities, each with its own set of streets and buildings. The streets and buildings in each city can be seen as a “category.” In each category, streets (corresponding to morphisms) connect buildings (corresponding to objects). The layout of streets and buildings may differ completely between cities.

A functor acts as a highly detailed map, not only showing the corresponding location of each building in another city but also explaining the corresponding route (street) in the other city. In other words, a functor is a rule that maps objects and morphisms in one category to objects and morphisms in another category while preserving relationships (maintaining the structure of the category). In comparison, a function only acts on objects, describing a simple mapping relationship between two sets.

Suppose there are two categories representing position and temperature. For example, $te(3) = 5$ indicates that the temperature at position 3 is 5. Here, positions and temperatures form a category, where objects are positions, and morphisms are mappings from positions to temperature values.

**These two concepts may sound complex, but they are not difficult. Predicates describe relationships or states between objects, while functors are mappings between categories. When we input a property value, we get the value of a new property in return.**

###### Logical Connectives

We often use propositional variables $P, Q, R$ to represent basic propositions or statements. Here are the five fundamental logical connectives:

- $\lnot$ Negation, representing “not”; $\lnot P$ means “not $P$.”
- $\land$ Conjunction, representing “and”; $P \land Q$ means “both $P$ and $Q$ hold.”
- $\lor$ Disjunction, representing “or”; $P \lor Q$ means “at least one of $P$ or $Q$ holds.”
- $\rightarrow$ Implication, representing “if…then…”; $P \rightarrow Q$ means “if $P$ holds, then $Q$ also holds.”
- $\leftrightarrow$ Biconditional, representing “if and only if”; $P \leftrightarrow Q$ means “$P$ holds if and only if $Q$ holds.”

In Carnap’s book, $\lnot$ is written as $\thicksim$, and $\land$ as $\cdot$ (note that in the next article, we’ll switch to the commonly used symbols). $\sum$ denotes the disjunction of multiple logical propositions, while $\prod$ denotes the conjunction of multiple logical propositions. $\equiv$ indicates the equivalence of two logical expressions.