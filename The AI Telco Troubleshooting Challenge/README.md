# 🔧 AI Telco Troubleshooting Challenge

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zindi Competition](https://img.shields.io/badge/Zindi-Competition-orange.svg)](https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge)

> **Track 3**: Can you build a specialised edge-cloud LLM to troubleshoot network faults?

Fine-tuning Qwen2.5-1.5B-Instruct to detect and explain unseen network failures using reasoning-enhanced supervised fine-tuning (R-SFT).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Approach](#approach)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Reproduction Steps](#reproduction-steps)
- [Results](#results)
- [Technical Report](#technical-report)
- [Acknowledgements](#acknowledgements)

---

## 🎯 Overview

This repository contains our solution for the **Zindi AI Telco Troubleshooting Challenge**. The goal is to enhance the accuracy of `Qwen2.5-1.5B-Instruct` when answering telco troubleshooting questions using the **telelogs** dataset.

### Key Objectives

- Fine-tune a lightweight LLM for edge-cloud deployment
- Maintain knowledge retention while specializing in network fault diagnosis
- Generate accurate root cause analysis in a single attempt (Pass@1 metric)

---

## 🚀 Approach

Our solution leverages **Reasoning-enhanced Supervised Fine-Tuning (R-SFT)**:

1. **Data Augmentation**: Combined Phase 1 test data (with ground truth) with training data
2. **Dataset Splitting**: Separated network troubleshooting vs. general knowledge questions
3. **Reasoning Trace Generation**: Used `Qwen2.5:7b-instruct` via IPEX-LLM accelerated Ollama to generate chain-of-thought reasoning
4. **Supervised Fine-Tuning**: Fine-tuned `Qwen2.5-1.5B-Instruct` with Unsloth 4-bit quantization
5. **Inference**: Model deployed on Kaggle for evaluation

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data      │ ──▶ │ Reasoning Traces │ ──▶ │ Fine-tuned LLM  │
│ (telelogs CSV)  │     │ (Teacher Model)  │     │ (Student Model) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 📁 Repository Structure

```
The AI Telco Troubleshooting Challenge/
├── 📄 README.md              # This file
├── 📄 REPORT.md              # Technical report (Markdown)
├── 📄 SETUP.md               # Environment documentation
├── 📄 requirements.txt       # Python dependencies
├── 📄 LICENSE                # MIT License
│
├── 📂 scripts/               # Data preparation pipeline
│   ├── augment_train_with_test_phase1.py
│   ├── split_phase2_test_network_faults_and_general_knowledge.py
│   ├── label_general_knowledge_dataset.py
│   ├── generate_reasoning_traces.py
│   └── prepare_data_for_fine_tuning.py
│
├── 📂 notebooks/
│   ├── kaggle/               # Kaggle notebooks
│   │   ├── 00-baseline-qwen2-5-1-5b-instruct.ipynb
│   │   └── notebookc4498523146-generate-reasoning-trace.ipynb
│   └── colab/                # Colab notebooks (if any)
│
├── 📂 data/
│   ├── raw/                  # Competition data (download from Zindi)
│   └── processed/            # Generated intermediate files
│
├── 📂 docs/                  # Reference documents & slides
├── 📂 img/                   # Images for documentation
├── 📂 report/                # LaTeX report for Overleaf
│   └── report.tex
│
└── 📂 submissions/           # Submission files
    ├── baseline/
    ├── sft/
    └── sft+rl/
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or 3.11
- (Optional) Intel Core Ultra with Arc iGPU for accelerated inference

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-telco-troubleshooting.git
cd ai-telco-troubleshooting

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download competition data from [Zindi](https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge/data)
2. Place files in `data/raw/`:
   - `train.csv`
   - `phase_1_test.csv`
   - `phase_1_test_truth.csv`
   - `phase_2_test.csv`
   - `SampleSubmission.csv`

---

## 🔄 Reproduction Steps

### Step 1: Data Preparation (Local)

Run the following scripts in order:

```bash
# 1. Augment training data with Phase 1 test + ground truth
python scripts/augment_train_with_test_phase1.py

# 2. Split Phase 2 test into network faults vs. general knowledge
python scripts/split_phase2_test_network_faults_and_general_knowledge.py

# 3. Label general knowledge questions
python scripts/label_general_knowledge_dataset.py

# 4. Generate reasoning traces (requires Ollama with qwen2.5:7b-instruct)
python scripts/generate_reasoning_traces.py

# 5. Prepare final SFT dataset
python scripts/prepare_data_for_fine_tuning.py
```

**Output**: `data/processed/sft_data.csv`

### Step 2: Fine-Tuning (Kaggle/Colab)

Use the notebooks in `notebooks/kaggle/` or `notebooks/colab/` for fine-tuning with Unsloth.

### Step 3: Inference

Generate predictions using the fine-tuned model. See `notebooks/kaggle/` for inference notebooks.

---

## 📊 Results

| Model                            | Pass@1 Accuracy | Notes              |
| -------------------------------- | --------------- | ------------------ |
| Baseline (Qwen2.5-1.5B-Instruct) | 0.1405          | Zero-shot          |
| R-SFT Fine-tuned                 | TBD             | Reasoning-enhanced |

---

## 📝 Technical Report

A comprehensive technical report is available in two formats:

- **Markdown**: [REPORT.md](REPORT.md)
- **LaTeX** (for Overleaf): [report/report.tex](report/report.tex)

The report covers:

- Methodology and approach
- Data privacy and compliance
- Model security risks
- Edge computing considerations
- Data governance

---

## 🙏 Acknowledgements

See [THANKS_TO.md](THANKS_TO.md) for full credits.

- **Competition Organizers**: Zindi, ITU, Politecnico di Milano
- **Tools**: Unsloth, Ollama, IPEX-LLM, Kaggle, Google Colab

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Kodjo Josué AYITEY (Yehoshua)  
**Date**: February 2026
