<p align="center">
  🚀 <strong>Reinforcement Learning Assignment 4</strong><br>
  <em>TD3, PPO, and SAC on Continuous-Action Environments</em><br>
  LunarLander-v3 & CarRacing-v3
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  </a>
  <a href="https://wandb.ai/">
    <img src="https://img.shields.io/badge/Weights_&_Biases-Tracking-orange?logo=weightsandbiases&logoColor=white">
  </a>
  <a href="https://huggingface.co/">
    <img src="https://img.shields.io/badge/HuggingFace-Model_Sharing-yellow?logo=huggingface&logoColor=white">
  </a>
  <a href="https://gymnasium.farama.org/">
    <img src="https://img.shields.io/badge/Gymnasium-Box2D-purple?logo=OpenAI&logoColor=white">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>



Ctrl + Shift + V  to view the readme 


# 📘 Overview

This project implements **three state-of-the-art model-free RL algorithms** — **TD3**, **PPO**, and **SAC** — trained on two continuous-control environments:

* **🌕 LunarLander-v3 (continuous=True)**
* **🏎️ CarRacing-v3 (continuous=True)**

The system includes:

* Full PyTorch implementations
* Custom environment wrappers
* Replay buffers & noise modules
* Weights & Biases experiment tracking
* Automatic video recording of agents
* Hugging Face leaderboard submission
* Reproducible training scripts

---

# 📂 Project Structure

```
rl-assignment4/
│
├── algorithms/
│   ├── td3/
│   │   ├── agent.py
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── config.py
│   ├── ppo/
│   │   ├── agent.py
│   │   ├── policy.py
│   │   ├── value_network.py
│   │   └── config.py
│   ├── sac/
│   │   ├── agent.py
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── temperature.py
│   │   └── config.py
│
├── common/
│   ├── replay_buffer.py
│   ├── noise.py
│   ├── normalization.py
│   ├── utils.py
│
├── envs/
│   ├── make_env.py
│   ├── wrappers_lunarlander.py
│   ├── wrappers_carracing.py
│
├── training/
│   ├── train_td3.py
│   ├── train_ppo.py
│   ├── train_sac.py
│   ├── eval_agent.py
│
├── huggingface/
│   ├── push_td3.py
│   ├── push_ppo.py
│   ├── push_sac.py
│   └── model_cards/
│       ├── td3_card.md
│       ├── ppo_card.md
│       └── sac_card.md
│
├── saved_models/
│   ├── td3/
│   ├── ppo/
│   └── sac/
│
├── reports/
│   ├── pdf/
│   └── figures/
│
├── videos/
│   ├── td3_lunarlander.mp4
│   └── sac_carracing.mp4
│
├── requirements.txt
└── README.md
```

---


### Dependencies include:

* PyTorch
* Gymnasium + Box2D
* NumPy
* Weights & Biases
* HuggingFace Hub
* OpenCV (optional)

---

# 🕹️ Supported Environments

## 🌕 **LunarLander-v3 (Continuous Mode)**

Action space:

```
[ main engine throttle, side engine throttle ]
```

## 🏎️ **CarRacing-v3**

Action space:

```
[ steering (-1..+1), gas (0..1), brake (0..1) ]
```

✔ High-dimensional observation (96×96 RGB)
✔ Custom wrappers included (grayscale, frame skip, resize)

---

# 🧠 Algorithms Implemented

### ✔ **TD3 — Twin Delayed Deep Deterministic Policy Gradient**

* Twin critics (Q1, Q2)
* Policy smoothing noise
* Delayed policy updates
* Target networks
* Replay buffer

### ✔ **PPO — Proximal Policy Optimization**

* Clipped objective
* Generalized Advantage Estimation (GAE)
* On-policy rollout buffer
* Mini-batch optimization

### ✔ **SAC — Soft Actor-Critic**

* Stochastic Gaussian actor
* Maximum entropy objective
* Automatic temperature tuning (α)
* Twin critics
* Replay buffer

---

# 🏋️ Training the Agents

### Train **TD3**

```bash
python training/train_td3.py --env lunarlander
python training/train_td3.py --env carracing
```

### Train **PPO**

```bash
python training/train_ppo.py --env lunarlander
python training/train_ppo.py --env carracing
```

### Train **SAC**

```bash
python training/train_sac.py --env lunarlander
python training/train_sac.py --env carracing
```

All training scripts include:

* Weights & Biases logging
* Automatic evaluation
* Checkpoint saving
* Video generation
* Configurable hyperparameters

---

# 🎥 Recording Videos of Trained Agents

Run evaluation with recording enabled:

```bash
python training/eval_agent.py --algo td3 --env lunarlander --record
```

Videos are saved into:

```
videos/
```

And automatically logged to W&B:

```python
wandb.log({"eval_video": wandb.Video(video_path)})
```

---

# 📊 Experiment Tracking — Weights & Biases

Training scripts log:

* Episode return
* Loss values (actor, critic, value function)
* Q1/Q2 critic estimates
* Entropy & α (for SAC)
* Evaluation metrics
* Videos of trained agents

### Creating Your W&B Report

1. Open your W&B project
2. Click **Reports → Create Report**
3. Add:

   * Learning curves
   * Comparison charts
   * Videos
4. Copy the public share link
5. Add it to your final PDF report

---

# 🤗 Hugging Face Leaderboard Submission

Each algorithm has its own upload script:

```bash
python huggingface/push_td3.py
python huggingface/push_ppo.py
python huggingface/push_sac.py
```

Uploads include:

* Model weights
* Model card (markdown)
* Evaluation video
* Training summary

You can see your submission under **hf.co/your-username**.

---

# ✔ Deliverables Checklist

| Deliverable                      | Status |
| -------------------------------- | ------ |
| GitHub repository with full code | ✅      |
| Video of trained agent           | ✅      |
| W&B experiment charts            | ✅      |
| W&B report link                  | ✅      |
| HuggingFace submission           | ✅      |
| Final PDF report                 | ✅      |

---

# 👥 Team Roles

| Member   | Algorithm | Environments           |
| -------- | --------- | ---------------------- |
| Maryam Habeb | TD3       | LunarLander, CarRacing |
| Aya Ayman | PPO       | LunarLander, CarRacing |
| Ziad Asar| SAC       | LunarLander, CarRacing |


