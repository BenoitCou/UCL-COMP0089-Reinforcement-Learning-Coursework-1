# COMP0089 — Reinforcement Learning • Coursework 2024 / 25  
MSc in Machine Learning, University College London

This repository contains everything the four assessed assignments of the **COMP0089 Reinforcement Learning** module.

| Path                                             | Description                                                                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `UCL_RL_part_1.ipynb`            | **Part I – Multi-armed bandits** &nbsp;•&nbsp; implementations of *UCB*, *ε-greedy* with time-varying schedules, and *softmax* agents; three experiments (stationary, sparse-reward, non-stationary) plus analytical questions on regret. |
| `UCL_RL_part_2.ipynb`           | **Part II – Tabular value-based RL** &nbsp;•&nbsp; TD(0) prediction, policy iteration and value-iteration on a Gridworld; proofs of Bellman-operator properties and empirical comparison of convergence speed. |
| `UCL_RL_part_3.ipynb`          | **Part III – Policy-gradient & actor-critic** &nbsp;•&nbsp; JAX/bsuite implementation of an actor-critic with shared network, advantage baseline, entropy regularisation; exploration of a preference-based alternative update rule. |
| `UCL_RL_part_4.ipynb`           | **Part IV – Off-policy multi-step returns** &nbsp;•&nbsp; full/weighted importance sampling, n-step TD with behaviour ≠ target policy, and theoretical analysis of a family of contraction-mapping updates. |
| `requirements.txt`                               | Python dependencies (numpy ≥ 1.25, matplotlib, seaborn, jupyter, gymnasium, bsuite, jax, tqdm).                                     |

---

## Quick-start

```bash
# 1. Clone the repo
git clone https://github.com/BenoitCou/UCL-COMP0089-Reinforcement-Learning-Coursework-1
cd UCL-COMP0089-Reinforcement-Learning-Coursework-1

# 2. Install the Python dependencies
pip install -r requirements.txt

# 3. Launch the notebooks
jupyter notebook UCL_RL_part_1.ipynb
# …and so on for parts 2–4
```

---

## Coursework overview  

**Part I**
- **Agents** — coded *UCB*, *ε-greedy* with linear decay, and a *softmax* agent; all share a generic incremental update for \(\hat q_t(a)\).
- **Experiments** — compared regret on stationary, sparse-reward and non-stationary bandits; analysed when each exploration strategy excels.
- **Theory questions** — derived regret bounds for UCB and explained the exploration–exploitation trade-off.

**Part II**
- **TD(0) Prediction** — Implemented on-policy TD learning for a random agent and verified empirical contraction toward \(v_\pi\).
- **Control** — Implemented policy iteration and value iteration agents for a grid world environment; measured iteration counts vs TD sweeps.
- **Bellman Proofs** — Demonstrated monotonicity and contraction of \(\mathcal{T}_\pi\) and derived the unique fixed point.
- **Grid World Environment** — Created a grid world with walls, goals, and empty spaces, where the agent can move in four directions and receive rewards.
- **Helper Functions** — Developed functions for running experiments, plotting values, and analyzing results.

**Part III**
- **Actor-Critic Implementation** — Developed an actor-critic agent using a neural network with JAX, featuring a stochastic softmax policy and TD(λ) for the critic. Implemented baseline-subtracted REINFORCE for policy updates.
- **Policy Gradient** — Computed stochastic estimates of the policy gradient using one-step transitions, incorporating a baseline to reduce variance.
- **Value Function Learning** — Implemented TD(0) updates for the value function, enabling simultaneous learning of policy and value estimates.
- **Optimization with Adam** — Applied the Adam optimization algorithm for adaptive learning rates, improving the stability and efficiency of training.
- **Epsilon-Greedy Policy** — Implemented an alternative agent using an epsilon-greedy policy for action selection, with a preference-based update rule for learning.
- **Comparison of Optimizers** — Conducted experiments comparing the performance of the Adam optimizer against standard SGD, demonstrating faster convergence and stability with Adam.
- **Analysis of Asymptotic Performance** — Evaluated the best achievable average returns for both the actor-critic and epsilon-greedy agents, discussing potential improvements and the role of preferences in action selection.


**Part IV**
- **Off-Policy Multi-Step Return Estimates** — Implemented various off-policy return functions, including full importance sampling, per-decision importance sampling, control variates, and adaptive bootstrapping. Analyzed their accuracy and effectiveness in reducing value error.
- **Temporal Difference Error Analysis** — Investigated the convergence properties and variance of updates for different temporal difference errors. Provided proofs and conditions under which these updates converge to optimal value functions.
- **Comparison of Return Estimates** — Evaluated and ranked different return estimates based on their mean squared value error. Discussed the implications of choosing the best return estimate and potential reasons for not always selecting the top-performing method.
- **Behavior Policy Analysis** — Examined the impact of different behavior policies on the convergence and variance of temporal difference learning. Proposed and justified conditions under which certain policies might be preferred.
- **Variance Reduction Techniques** — Explored methods to reduce variance in off-policy learning, including the use of control variates and adaptive bootstrapping. Discussed the trade-offs between bias and variance in these approaches.
- **Theoretical Insights** — Provided theoretical analysis and proofs related to the convergence of value functions under different conditions and behavior policies. Discussed the implications of the deadly triad in reinforcement learning.

---

## Marks obtained  

**Overall grade**: **84 / 100**

| Part | Score | Lecturer’s feedback (abridged) |
|------|-------|--------------------------------|
| I | 17 / 20 | |
| II | 20 / 25 | 1.1–1.2 great; 2.2 great. In Q2.8-1 mis-count (7 ≠ 8); Q2.8-2 missed insight \(v_\pi=0\); Q2.8-4 no empirical check; Q2.8-6 lacked eigenvalue view. |
| III | 23.5 / 25 | |
| IV | 23.5 / 30 | IS implementation bug (move `G × ρ` outside loop); include on-policy baseline in 1.2; in 2.1 list full contraction conditions; suggest ε-greedy in 2.2 (iii). |

---

## Repository structure  

```text
UCL-COMP0089-RL-Coursework/
├── UCL_RL_part_I.ipynb   # Bandits: UCB, ε-greedy, etc.
├── UCL_RL_part_II.ipynb  # Gridworld TD, policy/value iteration
├── UCL_RL_part_III.ipynb # Actor-critic & preference-gradient (JAX)
├── UCL_RL_part_IV.ipynb  # Off-policy returns, IS, contraction proofs
└── requirements.txt      # numpy, matplotlib, gymnasium, jax, …
```
