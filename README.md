# COMP0089 — Reinforcement Learning • Coursework 2024 / 25  
MSc in Machine Learning, University College London

This repository contains everything the four assessed assignments of the **COMP0089 Reinforcement Learning** module.

| Path                                             | Description                                                                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `UCL_RL_assignment_2025,_part_I.ipynb`            | **Part I – Multi-armed bandits** &nbsp;•&nbsp; implementations of *UCB*, *ε-greedy* with time-varying schedules, and *softmax* agents; three experiments (stationary, sparse-reward, non-stationary) plus analytical questions on regret. |
| `UCL_RL_assignment_2025,_part_II.ipynb`           | **Part II – Tabular value-based RL** &nbsp;•&nbsp; TD(0) prediction, policy iteration and value-iteration on a Gridworld; proofs of Bellman-operator properties and empirical comparison of convergence speed. |
| `UCL_RL_assignment_2025,_part_III.ipynb`          | **Part III – Policy-gradient & actor-critic** &nbsp;•&nbsp; JAX/bsuite implementation of an actor-critic with shared network, advantage baseline, entropy regularisation; exploration of a preference-based alternative update rule. |
| `UCL_RL_assignment_2025,_part IV.ipynb`           | **Part IV – Off-policy multi-step returns** &nbsp;•&nbsp; full/weighted importance sampling, n-step TD with behaviour ≠ target policy, and theoretical analysis of a family of contraction-mapping updates. |
| `requirements.txt`                               | Python dependencies (numpy ≥ 1.25, matplotlib, seaborn, jupyter, gymnasium, bsuite, jax, tqdm).                                     |

---

## Quick-start

```bash
# 1. Clone the repo
git clone https://github.com/BenoitCou/UCL-COMP0089-RL-Coursework
cd UCL-COMP0089-RL-Coursework

# 2. (Optional) create & activate a virtual environment
python -m venv .venv

# 3. Install the Python dependencies
pip install -r requirements.txt

# 4. Launch the notebooks
jupyter notebook UCL_RL_assignment_2025,_part_I.ipynb
# …and so on for parts II–IV
```

---

## Coursework overview  

**Part I – Multi-armed bandits**
- **Agents** — coded *UCB*, *ε-greedy* with linear decay, and a *softmax* agent; all share a generic incremental update for \(\hat q_t(a)\).
- **Experiments** — compared regret on Bernoulli, sparse-reward and non-stationary bandits; analysed when each exploration strategy excels.
- **Theory questions** — derived regret bounds for UCB and explained the exploration–exploitation trade-off.

**Part II – Tabular RL**
- **TD(0) prediction** — implemented on-policy TD for a random agent and verified empirical contraction toward \(v_\pi\).
- **Control** — policy-iteration and value-iteration agents for the 4 × 4 Gridworld; measured iteration counts vs TD sweeps.
- **Bellman proofs** — showed monotonicity & contraction of \(\mathcal T_\pi\) and derived the unique fixed point.

**Part III – Policy gradient**
- **Actor-critic** — stochastic softmax policy, critic with TD(λ), and baseline-subtracted REINFORCE estimate.
- **Preference-based update** — explored a shared-parameter formulation, demonstrated lower variance on CartPole-bsuite.
- **Analysis** — discussed entropy regularisation and the effect of learning-rate coupling between actor and critic.

**Part IV – Off-policy returns & new TD-like operators**
- **Importance sampling** — implemented ordinary IS, weighted IS and per-decision IS; compared bias/variance as function of horizon.
- **N-step returns** — unified view of eligibility-trace style updates under off-policy sampling.
- **Convergence proof** — proved a sufficient contraction condition for a family of linear operators; highlighted the role of the discount factor and behaviour-target support mismatch.

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
├── UCL_RL_assignment_2025,_part_I.ipynb   # Bandits: UCB, ε-greedy, etc.
├── UCL_RL_assignment_2025,_part_II.ipynb  # Gridworld TD, policy/value iteration
├── UCL_RL_assignment_2025,_part_III.ipynb # Actor-critic & preference-gradient (JAX)
├── UCL_RL_assignment_2025,_part_IV.ipynb  # Off-policy returns, IS, contraction proofs
└── requirements.txt                       # numpy, matplotlib, gymnasium, jax, …
```
