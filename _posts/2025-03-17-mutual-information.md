---
layout: post
title: "What is Mutual Information?"
date: 2025-03-17
categories: infotheory
permalink: /infotheory/mutual-information/
author: "Joonkyu Min"
---


**Mutual information** is a measure of information of one random variable containing about the another.
It can be thought as correlation between two distribution.

Mutual information is defined by the **KL divergence** between the joint distribution and the product distribution.

$$
I(X;Y)=\sum_{x,y}p(x,y)\log {\frac{p(x,y)}{p(x)p(y)}}=D(p(x,y)\parallel p(x)p(y))
$$

If the two random variable is independent, joint distribution equals product distribution, we can easily state that mutual information is 0.

What is the relation with **entropy**?
Mutual information is the amount of information of $X$ that is reduced when conditioned by another random variable $Y$.

$$
\begin{align}
I(X;Y) & =\sum_{x,y}p(x,y)\log {\frac{p(x,y)}{p(x)p(y)}} \\
 & =\sum_{x,y}p(x,y)\log {\frac{p(x|y)}{p(x)}} \\
 & = -\mathbb{E}_{p}\log{p(x)} + \mathbb{E}_p\log {p(x|y)} \\
 & = H(X)-H(X|Y)
\end{align}
$$

This gives intuition that conditioning reduces entropy, 
$H(X|Y)\leq H(X)$.

Since $H(X\mid Y)=H(X,Y)-H(Y)$, mutual information can be also expressed as a symmetric form.

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
$$

Also, $I(X;X)=H(X)$, which once again shows why entropy can be interpreted as self information.

Mutual entropy also satisfies chain rule, expressed as follows.

$$
I(X_{1},X_{2},\dots,X_{n};Y)=\sum_{i=1}^nI(X_{i};Y|X_{1},\dots,X_{i-1})
$$


<!-- **Data processing inequality** -->

One important result of mutual information is the **data processing inequality**.

In a Markov chain 

$$
X\to Y\to Z
$$

data processing inequality says $I(X;Y)\ge I(X;Z)$, which means that as the data is processed, mutual information is lost.

$$
\begin{align}
I(X;Y,Z) & =I(X;Y)+I(X;Z|Y)=I(X;Y) \\
I(X;Z,Y) & =I(X;Z)+I(X;Y|Z)\ge I(X;Z) \\ \\
\therefore I(X;Y) & \ge I(X;Z)
\end{align}
$$
