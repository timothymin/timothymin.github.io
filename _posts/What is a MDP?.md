---
layout: default
title: "What is a MDP?"
categories: study
permalink: /study/
---

MDP is Markov Decision process, which is a sequential decision problem modeling that holds a Markov property.

The basic components of MDP is 
- Set of **states** $S$
- Set of **actions** $A$
- **Transition model** $P(s'|s,a)$
- **Reward function** $R(s, a, s')$
- **Discount factor**: $\gamma ∈[0,1)$ gives smaller weights to future rewards

The states should be fully observable in MDP.
If the states are not fully observable, it is called POMDP, which can be modified into MDP.

 **Policy** $\pi$ is a mapping from states to actions
  $\pi:S\rightarrow \{a|a\in Actions(s), s\in S\}$ 

 We want to find the optimal policy $\pi^*$ that maximizes the expected sum of rewards


Objective for MDP
- **Return**: $G_{t} = \sum^\infty_{i=0}\gamma^i R_{t+i+1}$, where $R_{t} = R(s_{t},a_{t},s_{t+1})$
- discounted utility (to avoid infinite utility value)
$$
U([s_{0},s_{1}, s_{2}, \dots])=\sum\gamma^t R(s_{t})\le \frac{R_{max}}{1-\gamma}
$$
$$
\pi_{s}^*=\arg \max_{\pi}U^\pi(s)
$$

