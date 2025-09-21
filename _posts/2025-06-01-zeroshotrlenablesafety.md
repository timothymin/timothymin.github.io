---
layout: post
title: "Can Zero-shot RL enable test time safety?"
date: 2025-06-01 +00:00
permalink: /post/zeroshotrlenablesafety/
author: "Joonkyu Min"
---

To develop a truly generalized agent, it is essential to enable it to perform well on multiple unseen tasks, without training a separate agent for each task. In standard reinforcement learning (RL), an agent trained on a single, fixed reward function defined a specific task, typically lacks the ability to generalize beyond that task. 
The zero-shot reinforcement learning addresses this challenge by training agents without access to explicit reward signals, and aiming to produce policies that can be conditioned by any given rewards instantly at test time. This capability is particularly valuable in offline RL settings, where agents are pretrained on large-scale datasets that does not contain reward labels. 

Zero-shot RL shows potential to develop general- purpose, reusable agents that can be directly deployed in real-world environments, without requiring additional data collection, which is often expensive or dangerous.
Recent advancements in zero-shot RL have trained the decouple environment dynamics from the reward function, enabling generalization across tasks. However, a critical limitation of existing zero-shot RL approaches is their lack of safety guarantees, which is crucial for actual deployment. 
Unlike standard safe RL settings, where unsafe behaviors can be corrected during training, zero-shot RL agents should be able to immediately satisfy the post-defined safety constraints when executing new policies in the environment. 

This raises a key question: If zero-shot RL problem is to adapt to any rewards, can it also adapt to any safety constraints?


To address this question, we propose \textbf{FB-safe}, a zero-shot safety method that extends FB representations\cite{touati2021fb} to incorporate safety. 
Our approach enables agents to dynamically adapt to safety constraints at test time by leveraging generalized Q-functions that span both reward and cost signals.
This eliminates the need for further exploration or fine-tuning and allows for safe, efficient deployment in offline safe RL scenarios.
We evaluate FB-safe against prior state-of-the-art methods to demonstrate its effectiveness in balancing task performance and safety, which showed competitive performance.
Our approach shows potential of zero-shot RL problem can be extended to a zero-shot safety problem, proposing a novel perspective on how to approach safety in RL without task-specific training.



**Constrained Markov Decision Process (CMDP)**
In order to incorporate safety into zero-shot RL problem, we consider a modified form of Constrained Markov Decision Process (CMDP), which is typically used in safe RL.

At train phase, reward free MDP defined by tuple $(S, A, P, \gamma)$ is given, where where $S$ is the state space, $A$ is the action space, $P: S\times A \times S \to \mathbb{R}$ is the transition probability, and $\gamma\in [0,1)$ is the discount factor.

At test phase, reward function $R: S \to \mathbb{R}$, and cost function $C: S \to \mathbb{R}_{\ge0}$ that penalizes unsafe behavior, are defined to form a CMDP $(S, A, P, R, C, \gamma)$.
The main goal for the policy in CMDP is to maximize expected return while ensuring that the expected cumulative cost satisfies an upper bound $\epsilon$.
\begin{equation}
\pi^*_{R,C} = \arg\max_\pi \mathbb{E}^\pi[\sum R(s_t)],\textbf{ s.t. } \mathbb{E}^\pi[\sum C(s, a)] \le \epsilon
\end{equation}

**FB representation**
For the training phase, we utilize prior method of zero-shot RL, FB framework\cite{touati2021fb}. 
\begin{equation}
\begin{split}
M^\pi_z(s_0, a_0, s) &= \sum_{t\ge0} \gamma^t \Pr(s_{t+1} = s \mid s_0, a_0, \pi),
\end{split}
\end{equation}
It approximates successor measure in finite-dimensional space by using two parametric functions: forward mapping $F_{z}^T: S\times A\to \mathbb{R}^d$, and backward mapping $B: \mathbb{R}^d\to S$, such that the successor measure is represented as 
\begin{equation}
\begin{split}
M^\pi_z(s_0, a_0, s')=F(s_0,a_0,z)^TB(s')\rho(ds'),\\
\end{split}
\end{equation}
where $\rho$ is the data distribution.
For any reward function $R$, we can estimate the latent vector of reward by $z_R =\mathbb{E}_\rho[R(s)B(s)]$, using small amount of samples of given data.
Theoretical results have shown the optimal $Q$-function and optimal policy of the reward parameterized by $z_R$ can be derived by 
\begin{equation}
    \begin{split}
        Q_R^* &= F(s,a,z_R)^Tz_R, \\
        \pi_{R}^*(s)&=\arg\max_a  F(s,a,z_R)^Tz_R.
    \end{split}
\end{equation}

Training the forward, backward parametric functions are done by TD learning using the following loss,

\begin{equation}
\begin{split}
L_{\text{FB}} =& \mathbb{E}_{(s_t,a_t,s_{t+1})\sim\rho}\bigg[ \bigg( F(s_t, a_t, z)^\top B(s') \\ &- \gamma \, \bar{F}(s_{t+1}, \pi_z(s_{t+1}), z)^\top \bar{B}(s') \bigg)^2 \bigg] \\
&- 2 \mathbb{E}_{\rho, z}\left[ F(s_t, a_t, z)^\top B(s_{t+1}) \right],
\end{split}
\end{equation}

where the $\bar{F}, \bar{B}$ is the target network, and $\pi_z$ is the $z$-parameterized deterministic actor that is jointly trained just like DDPG\cite{lillicrap2015ddpg}.

Also, introducing an additional conservative learning term into this framework enhances the offline RL performance\cite{jeen2024zerolow}.
\begin{equation}
\begin{split}
L_{VC} = 
&\mathbb{E}_{{s \sim\rho, a \sim \mu(\cdot|s)}}
\left[ F(s, a, z)^\top z \right]\\
&-\mathbb{E}_{{(s,a)\sim\rho}}
\left[ F(s, a, z)^\top z \right] - H(\mu),
\end{split}
\end{equation}
where $\mu$ is the distribution of policy at each state, and $H(\mu)$ is approximated by log-sum exponential of $Q_z$.

\section{Method}

The ideal test-time objective can be written in a unified form as follows:
\begin{equation}
    \begin{split}
&\max_\pi \mathbb{E}^{\pi} \bigg[ \sum_{t=0}^{\infty} \gamma^t R(s_t)- \lambda\max(\sum_{t=0}^{\infty}\gamma^t C(s_t) - \epsilon,  0) \bigg].
\end{split}
\end{equation}
Instead, we aim to maximize the following objective that is easier to compute, 
\begin{equation}
    \begin{split}
\begin{cases}
\mathbb{E}^{\pi} \big[ \sum \gamma^t (R(s_t)- \lambda C(s_t))\big],  &\text{if} \max \mathbb{E}^\pi[\sum\gamma^t C(s_t)] \ge \epsilon \\
\mathbb{E}^{\pi} \big[ \sum \gamma^t (R(s_t))\big],  &\text{if} \max \mathbb{E}^\pi[\sum\gamma^t C(s_t)] < \epsilon
\end{cases}
\end{split}
\end{equation}
This can be interpreted as follows:
If the current state-action is expected to not exceed cost threshold at worst, we simply maximize the reward. But if the current state-action is expected to exceed cost threshold at worst, we maximize the reward, while minimizing the reward. 

This can be written again with Q-functions.
In CMDP, the $Q$-function is defined as
\begin{equation}
Q^\pi_R(s, a) = \mathbb{E}^{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t) \,\middle|\, s_0 = s, a_0 = a \right],
\end{equation}
and also the $Q$-function of the cost can be also defined, which is expressed as
\begin{equation}
Q^\pi_C(s, a) = \mathbb{E}^{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t C(s_t) \,\middle|\, s_0 = s, a_0 = a \right].
\end{equation}

The $\mathbb{E}^{\pi} \big[ \sum \gamma^t (R(s_t)- \lambda C(s_t))\big]$ can be considered as a Q-function when regarding $R-\lambda C$ as the reward function, which is a cost-penalized reward.
Also, $\max\mathbb{E}^\pi[\sum\gamma^t C(s_t)] $ can be considered as the optimal Q-function of the cost, which regards cost $C$ as a reward.
% Q^\pi_C(s, a) = \mathbb{E}^{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t C(s_t) \,\middle|\, s_0 = s, a_0 = a \right].

Therefore, the objective is expressed as
\begin{equation}
    \begin{split}
    \max_\pi \begin{cases}
Q^\pi_{R-\lambda C}(s,a),  & \text{if } Q^*_C(s,a) \ge \epsilon \\
Q^\pi_{R}(s,a),  & \text{if } Q^*_C(s,a) < \epsilon
\end{cases}
\end{split}
\end{equation}
Derived from this proximal objective, the proposed \textbf{FB-safe} policy is a composite policy of the following two optimal policies $\pi^*_{R-\lambda C}$ and $\pi^*_R$:
\begin{equation}
\begin{split}
    \pi^*_{R-\lambda C} &= \arg\max_\pi Q^\pi_{R-\lambda C}(s,a), \\
        \pi^*_{R} &= \arg\max_\pi Q^\pi_R(s,a).
\end{split}
\end{equation}
Given cost and reward samples, we can compute the latent vectors
\begin{equation}
\begin{split}
z_C&=\mathbb{E}_\rho[C(s)B(s)],\\ 
z_{R-\lambda C}&=\mathbb{E}_\rho[(R(s)-\lambda C(s))B(s)]
\end{split}
\end{equation}
and using the pretrained FB-framework, we can compute  $\pi^*_{R-\lambda C}$ and $\pi^*_R$, as well as $Q^*_C$.
The final \textbf{FB-safe} policy can be derived as
\begin{equation}
\pi_{safe} = 
\begin{cases}
\pi^*_{R} & \text{if } Q^*_C \le \epsilon \\
\pi^*_{{R-\lambda C}} & \text{if } Q^*_C > \epsilon
\end{cases}
\end{equation}
The $\pi_{R-\lambda C}$ can be also thought as a recovery policy, which leads to the safe region when it gets risky.
In this method, the choice of $\lambda$ is crucial. In order to balance maximizing reward and minimizing cost, $\lambda=\mathbb{E}_\rho[R(s)]/\mathbb{E}_\rho[C(s)]$ was used in practice.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=1.0\columnwidth]{Can Zero-shot RL ensure safety?.key.pdf}
  \caption{\textbf{Diagram of FB-safe}. Figure shows the process of FB-safe policy, at test time. Given samples of reward and cost, penalized reward is computed. Then, using the pretrained FB-framework, safe recovery policy $\pi_{R-\lambda C}$, policy $\pi_{R}$, and the safety critic function $Q_C^*$ is derived. The final action is selected at execution depending on the safety critic.}
  \label{fig:my_label}
\end{figure}

\begin{table*}[!h]
\centering
\caption{Normalized Reward and Cost comparison across methods on various tasks of BulletGym Environment. \textbf{Bold} indicates safe policy. Averaged for 3 different seeds on 20 episodes, $\epsilon$ of [10, 20, 40].}
\label{tab:reward_cost_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cc|cc|cc|cc|cc|cc}
\toprule
& \multicolumn{2}{c|}{\textbf{BC-safe}} & \multicolumn{2}{c|}{\textbf{CDT}} & \multicolumn{2}{c|}{\textbf{COptiDICE}} & \multicolumn{2}{c|}{\textbf{CPQ}} & \multicolumn{2}{c|}{\textbf{FB}} & \multicolumn{2}{c}{\textbf{FB-safe}} \\
\textbf{Task} & reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$\\
\midrule
CarCircle   & 0.50 & \textbf{0.84} & 0.75 &\textbf{ 0.95} & 0.49 & 3.14 & 0.71 &\textbf{ 0.33} & 0.87 & 5.27 & 0.45 & \textbf{0.27} \\
DroneCircle & 0.56 & \textbf{0.57} & 0.63 & \textbf{0.98} & 0.26 & 1.02 & -0.20 & 1.28 & 0.57 & \textbf{0.99} & 0.26 & \textbf{0.18} \\
DroneRun    & 0.28 & \textbf{0.74} & 0.63 & \textbf{0.79} & 0.67 & 4.15 & 0.33 & 3.52 & 1.00 & 8.17 & 0.58 & 3.16 \\
AntRun      & 0.65 & 1.09 & 0.72 &\textbf{ 0.91} & 0.61 & \textbf{0.94} & 0.03 & \textbf{0.02} & 0.68 & 4.74 & 0.44 & \textbf{1.00} \\
\bottomrule
\end{tabular}
}
\end{table*}


We conduct our experiments on the DSRL benchmark \cite{liu2024dsrl}, which provides offline datasets tailored for safe reinforcement learning. Specifically, we use the BulletSafetyGym \cite{Gronauer2022BulletSafetyGym} from the benchmark.
We compare our method against recent offline safe RL baselines, including BC-safe \cite{liu2024dsrl}, CPQ \cite{xu2022cpq}, COptiDICE \cite{lee2022coptidice}, CDT \cite{liu2023cdt}.
We note that these offline safe RL baselines are trained given the reward and cost, while our method is trained without access to reward and cost, as in zero-shot RL setting.
Table~\ref{tab:reward_cost_comparison} shows the normalized reward and cost that is proposed in DSRL, which cost value indicates safe when it is below 1. The reward is normalized by the maximum reward value of each environment.
We follow the DSRL benchmark evaluation, to run 3 different seeds for each method, and each 20 episodes of threshold $\epsilon$ of [10, 20, 40]. The table shows the average result across all experiments.

Our method achieves a competitive performance of trade-off between reward maximization and cost minimization across various tasks. 
The proposed FB-safe policy significantly reduces cost compared to the vanilla FB policy, demonstrating the effectiveness of our safe policy layer even in zero-shot manner.

It is particularly remarkable that FB-safe maintains favorable performance in tasks such as DroneRun and DroneRun, where cost-augmented policies such as CPQ often suffer from poor generalization. 
FB-safe method shows competitive or better results with baselines such as COptiDICE and CPQ, while CDT, which is a transformer based approach shows good generalization resulting an outperforming result.

Moreover, we observe that the choice of the hyperparameter $\lambda$, which governs the trade-off between reward and safety during policy selection, plays a critical role in adapting the agent to different operational contexts. As shown in the Table~\ref{tab:reward_cost_comparison_lambda}, a small $\lambda$ doesn't prevent the risky behavior, while a larger $\lambda$ achieve safer results, but results in a lower reward.
Effectively balancing this trade-off through an appropriate selection or adaptation of $\lambda$ remains an important direction for future work.

\begin{table}[!h]
\centering
\caption{Normalized Reward and Cost comparison across $\lambda$ on CarCircle Environment}
\label{tab:reward_cost_comparison_lambda}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l|cc|cc|cc|cc}
\toprule
\textbf{$\lambda$} & \multicolumn{2}{c|}{\textbf{1}} & \multicolumn{2}{c|}{\textbf{2}} & \multicolumn{2}{c|}{\textbf{4}} & \multicolumn{2}{c}{\textbf{8}} \\
& reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$& reward $\uparrow$& cost $\downarrow$& 
\midrule
\textbf{CarCircle}   & 0.74 & 4.17 & 0.69 & 3.07 & 0.58 & \textbf{0.84} & 0.42 & \textbf{0.24} \\
\bottomrule
\end{tabular}
}
\end{table}


**Conclusion**

In this work, we propose a zero-shot safe RL framework that leverages Forward-Backward (FB) representations to enable generalization across tasks without relying on reward or cost signals during training.
We introduce FB-safe, an extension of FB representations with a safety-aware policy selection mechanism, which effectively balances reward and cost in offline safe RL environments.

Our results on the DSRL benchmark demonstrate that FB-safe matches or outperforms existing offline safe RL baselines in terms of safety-cost reduction, even though it operates in a more challenging zero-shot setting. These findings suggest that zero shot RL problem, when combined with implicit cost estimation, provide a powerful and generalizable mechanism for safe policy deployment.

This approach is highly remarkable because it can be used in situations where costs are dynamically defined. 
Therefore, we further plan to extend our method to handle pixel-based state representations and incorporate safety constraints expressed in natural language. 
This can enable even greater generalization and explainability in real-world deployment scenarios, where explicit safety signals such as collisions or hard-coded failure states are hard to detect, or may not capture the full complexity of human-defined safety.
This approach enables agents to interpret and satisfy the nuanced safety specifications, providing more flexible and generalizable framework for safe reinforcement learning.

One limitation of our approach is that $\lambda$ has high effect on performance, yet we use it as a fixed hyperparameter. Developing a tunable, adaptable method to find $\lambda$ to balance the performance might be a direction to explore in the future.

By addressing these directions, we believe our approach can serve as a robust foundation for safe, generalizable, and interpretable reinforcement learning systems, especially in environments with dynamic and complex safety such as robotics and autonomous navigation.

