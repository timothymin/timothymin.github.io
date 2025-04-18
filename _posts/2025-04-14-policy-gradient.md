---
layout: post
title: "What is Policy Gradient?"
date: 2025-04-14
categories: study
permalink: /study/policy-gradient/
author: "Joonkyu Min"
---

Policy gradient is a method of RL that optimize the policy directly, by computing the gradient of the objective function w.r.t. the policy function parameters.

**Likelihood Ratio policy gradient** is the foundation idea of policy gradient. 
Consider the likelihood of each trajectory under policy parameter $\theta$ as $P(\tau;\theta)$.
We want to maximize the following objective.

$$
U(\theta)=\mathbb{E}\left[ \sum R(s_{t},a_{t});\pi_{\theta} \right]=\sum_{\tau}P(\tau;\theta)R(\tau)
$$

The gradient of the objective w.r.t. $\theta$ becomes the expectation form of gradient of log likelihood.

$$
\begin{align}
\nabla_{\theta}U(\theta) & = \nabla_{\theta}\sum_{\tau}P(\tau;\theta)R(\tau)  \\
 & =\sum_{\tau}\nabla_{\theta}P(\tau;\theta)R(\tau) \\
& = \sum_{\tau}P(\tau;\theta) \frac{\nabla_{\theta}P(\tau;\theta)}{P(\tau;\theta)}R(\tau) \\
 & = \sum_{\tau}P(\tau;\theta) \nabla_{\theta}\log P(\tau;\theta)R(\tau) \\ 
 & =\mathbb{E}[\nabla_{\theta}\log P(\tau;\theta)R(\tau)]
\end{align}
$$

This intuitively updates the parameter to the direction of pushing up the likelihood of high rewards.

The gradient of likelihood actually becomes gradient of the policy because

$$
\begin{align}
\log P(\tau;\theta) & =\log\left[ \prod P(s_{t+1}|s_{t},a_{t})\pi_{\theta}(a_{t}|s_{t}) \right] \\
 & =\sum_{t}\log P(s_{t+1}|s_{t},a_{t}) + \sum_{t} \log\pi_{\theta}(a_{t}|s_{t})
\end{align}
$$

and the first term has nothing to do with $\theta$.

Thus, we can compute the unbiased estimate of the gradient by this way.

$$
\begin{align}
\nabla_{\theta}U(\theta)
 & =\mathbb{E}\left[ \left(\sum_{t} \nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t})\right)R(\tau) \right]
\end{align}
$$

This version of gradient has high variance.
We can reduce the variance in two ways.
1. remove past rewards

$$
\begin{align}
\nabla_{\theta}U(\theta)
 & =\mathbb{E}\left[ \left(\sum_{t} \nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t})\sum_{t'}\gamma^{t'} r_{t'} \right) \right] \\
 & =\mathbb{E}\left[ \sum_{t} \nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t})\left(\sum_{t'=0}^{t-1}\gamma^{t'} r_{t'}+\gamma^t\sum_{t'=t}\gamma^{t'-t} r_{t'} \right) \right] \\ 
 & =\mathbb{E}\left[ \sum_{t} \nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t})\gamma^tG_{t} \right] \\ 
\end{align}
$$

The past rewards is not influenced by the random action at time $t$, which makes the expected derivative 0.

2. subtract a baseline function that doesn't depend on action

$$
\begin{align}
\nabla_{\theta}U(\theta)
 & =\mathbb{E}\left[\sum_{t} \nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t})\gamma^t(G_{t}-b(s_{t})) \right]
\end{align}
$$

Using the value function as a baseline is a intuitive choice.
It updates the parameter to the direction of pushing up the likelihood of rewards higher than the average, and push down the likelihood of rewards lower than the average.

This vanilla policy gradient is also called REINFORCE algorithm.
