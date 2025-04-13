---
layout: post
title: "What is a MDP?"
date: 2025-04-13
categories: study
permalink: /study/what-is-a-mdp/
author: "Joonkyu Min"
---

MDP is Markov Decision process, which is a sequential decision problem modeling that holds a Markov property.

The basic components of MDP is 
- $S$: **states**
- $A$: **actions**
- $P(s'|s,a)$: **transition model**
- $R(s, a, s')$: **reward**
- $\gamma \in[0,1)$: **discount factor** that gives smaller weights to future rewards

The states should be fully observable in MDP.

The problem setting of which the states are not fully observable is called POMDP, which can be modified into MDP.

In MDP, we can define a **Policy** $\pi:S\rightarrow \{a|a\in Actions(s), s\in S\}$, which is a mapping from states to actions.
What we want to do in MDP is to find the optimal policy $\pi^*$ that maximizes the expected sum of rewards.

That is, in a mathematical formation, we want to maximize the main objective **Return** for MDP.
It can be also interpreted as dicounted utility function.
$$G_{t} = \sum^\infty_{i=0}\gamma^i R_{t+i+1}$, where $R_{t} = R(s_{t},a_{t},s_{t+1})$$
$$
U([s_{0},s_{1}, s_{2}, \dots])=\sum\gamma^t R(s_{t})\le \frac{R_{max}}{1-\gamma}
$$
$$
\pi_{s}^*=\arg \max_{\pi}U^\pi(s)
$$

