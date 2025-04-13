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
- **state**: $S$
- **action**: $A$
- **transition model**: $P(s'\mid s,a)$
- **reward**: $r(s, a, s')$
- **discount factor**: $\gamma \in[0,1)$, it gives smaller weights to future rewards

The states should be fully observable in MDP.

The problem setting of which the states are not fully observable is called POMDP, which can be modified into MDP.

In MDP, we can define a **policy** 
$\pi:S\rightarrow \{a\mid a\in A, s\in S\\}$, 
which is a mapping from states to actions.
What we want to do in MDP is to find the optimal policy $\pi^*$ that maximizes the expected sum of rewards.

That is, in a mathematical formation, we want to maximize the following dicounted utility function objective.

$
G^{\pi}_{t} = \mathbb{E}^\pi \sum^\infty_{i=0}\gamma^i r_{t+i+1} \le \frac{\sup r}{1-\gamma}
$, 

where $r_{t} = r(s_{t},a_{t},s_{t+1})$

$
\pi_{s}^*=\arg \max_{\pi}G^\pi(s)
$

