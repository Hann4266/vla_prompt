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



