---
layout: post
title: "Can Zero-shot RL enable test time safety?"
date: 2025-06-01
permalink: /post/zeroshotrlenablesafety/
author: "Joonkyu Min"
---

To develop a truly generalized agent, it is essential to enable it to perform well on multiple unseen tasks, without training a separate agent for each task. In standard reinforcement learning (RL), an agent trained on a single, fixed reward function defined a specific task, typically lacks the ability to generalize beyond that task. 
The zero-shot reinforcement learning [1] addresses this challenge by training agents without access to explicit reward signals, and aiming to produce policies that can be conditioned by any given rewards instantly at test time. This capability is particularly valuable in offline RL settings, where agents are pretrained on large-scale datasets that does not contain reward labels. 
Zero-shot RL shows potential to develop general- purpose, reusable agents that can be directly deployed in real-world environments, without requiring additional data collection, which is often expensive or dangerous.
Recent advancements in zero-shot RL have trained the decouple environment dynamics from the reward function, enabling generalization across tasks. However, a critical limitation of existing zero-shot RL approaches is their lack of safety guarantees, which is crucial for actual deployment. 
Unlike standard safe RL settings, where unsafe behaviors can be corrected during training, zero-shot RL agents should be able to immediately satisfy the post-defined safety constraints when executing new policies in the environment. 
This raises a key question: If zero-shot RL problem is to adapt to any rewards, can it also adapt to any safety constraints?


