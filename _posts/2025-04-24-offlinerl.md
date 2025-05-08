---
layout: post
title: "What is Offline RL?"
date: 2025-04-24
categories: RL
permalink: /RL/offlinerl/
author: "Joonkyu Min"
---

**Offline reinforcement learning**, also known as batch reinforcement learning is a method in between standard online RL and data-driven machine learning.
Online RL agents require active data collection during training, so the data is likely to be limited in real world scenarios, leading to poor generalization.
In contrast, (self) supervised machine learning uses large and diverse datasets to train the models, allowing them to scale easily and generalize well.

Offline RL aims to learn policies from a fixed dataset without any interaction with the environment. 
Although off-policy methods, which update policies using old data stored in a replay buffer, also utilize data not collected concurrently, they don't perform well in offline settings due to extrapolation error when the agent encounters out-of-distribution states.

The main challenge of offline RL is **distribution shift**. 
Since the agent is trained on a static dataset, often collected under an expert policy or a separate exploration policy, the function approximation are not reliable especially for states that are not represented well in the dataset.
This often leads to overestimation in value estimates, which causes problems in Q-learning or actor-critic methods.

To mitigate this, a widely used method is Conservative Q-Learning (CQL) which penalizes overestimated Q-values by enforcing conservatism in value estimation.

