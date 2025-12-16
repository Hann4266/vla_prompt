# Two-Stage VLA Prompting Pipeline

**Vision-Language-Action (VLA) for Autonomous Driving with Chain-of-Causation (CoC) Annotation.**

This repository implements a two-stage pipeline designed to reason about autonomous driving scenarios. It utilizes a Large Language Model (LLM) to extract critical visual information from history frames and infer driving decisions based on future trajectories.

---

## 📂 Repository Structure

```text
vla_prompt/
|
├── llm_nividia_annotation.py    # Main script executing the pipeline
|
├── prompt_stage_1/              # Stage I Prompt Engineering
│   ├── STAGE1_SYSTEM_INSTRUCTION
│   └── STAGE1_USER_TASK
|
├── prompt_stage_2/              # Stage II Prompt Engineering
│   ├── STAGE2_SYSTEM_INSTRUCTION
│   └── STAGE2_USER_TASK
|
├── data_set_1/                  # Input Data Directory 1
└── data_set_2/                  # Input Data Directory 2
```

## Configure the script

Open `llm_nividia_annotation.py` and edit these variables:

```python
FOLDER   = "/path/to/Annotation With NividaPrompting/data_set_1"
MODEL    = "glm-v"
BASE_URL = "https://ellm.nrp-nautilus.io/v1"
API_KEY  = "YOUR_API_KEY"

N_FUTURE_FRAMES = 10

# insert the overall navigation goal here
NAV_GOAL = "goal of your data"
```

Notes:
- `FOLDER` = the dataset folder you want to run on (example: `data_set_1`)
- `N_FUTURE_FRAMES` = last N images are treated as **future frames**; everything before that is **history**
- `NAV_GOAL` is injected into Stage II by replacing `__INSERT_NAVIGATION_GOAL_HERE__` inside `prompt_stage_2/STAGE2_USER_TASK`



