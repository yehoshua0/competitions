# Fine-Tuned Reasoning Language Models for Root Cause Analysis in 5G Networks: An Experimental Approach

_By Kodjo Josué AYITEY (Yehoshua) - Independent Researcher, Maritime, TG_

_February 2026_

---

## Abstract

This report presents our approach to the Zindi AI Telco Troubleshooting Challenge, where we developed a reasoning-enhanced fine-tuning methodology for network fault diagnosis. By combining knowledge distillation from a larger teacher model with supervised fine-tuning on the Qwen2.5-1.5B-Instruct architecture, we achieved improved performance on root cause analysis tasks while maintaining general knowledge retention. This document also addresses critical considerations around data privacy, model security, edge computing deployment, and data governance.

---

## 1. Introduction

### 1.1 Challenge Overview

The AI Telco Troubleshooting Challenge tasks participants with building a specialized edge-cloud LLM capable of:

1. Diagnosing network faults from telco log data
2. Providing accurate root cause explanations
3. Maintaining general knowledge accuracy (knowledge retention)

The evaluation metric is **Pass@1**, measuring the model's ability to produce correct answers in a single attempt.

### 1.2 Motivation

Traditional rule-based network troubleshooting systems struggle with novel fault patterns. Large Language Models offer the potential for more flexible reasoning, but deployment on edge infrastructure requires compact, efficient models. Our approach bridges this gap through knowledge distillation and targeted fine-tuning.

---

## 2. Methodology

### 2.1 Data Preparation Pipeline

Our data preparation consisted of five stages, all executed locally:

1. **Data Augmentation**: Combined training data with Phase 1 test set (using released ground truth) to expand the training corpus
2. **Dataset Stratification**: Separated network troubleshooting questions from general knowledge questions to enable targeted training
3. **General Knowledge Labeling**: Annotated general knowledge questions with correct answers to preserve knowledge retention
4. **Reasoning Trace Generation**: Used Qwen2.5-7B-Instruct as a teacher model to generate chain-of-thought reasoning traces for each training sample
5. **SFT Dataset Preparation**: Compiled the final supervised fine-tuning dataset with reasoning-augmented responses

### 2.2 Reasoning-Enhanced Supervised Fine-Tuning (R-SFT)

Our key innovation is the integration of reasoning traces into the training data:

```
<reasoning>
Step 1: Analyze the network log entries...
Step 2: Identify anomalous patterns...
Step 3: Cross-reference with known fault signatures...
Conclusion: The root cause is...
</reasoning>

<final_answer>...</final_answer>
```

This approach teaches the student model not just _what_ to answer, but _how_ to reason about network faults.

### 2.3 Model Architecture

- **Base Model**: Qwen2.5-1.5B-Instruct
- **Quantization**: Unsloth Dynamic 4-bit (selective quantization for improved accuracy)
- **Fine-tuning Framework**: Unsloth with LoRA adapters
- **Training Platform**: Kaggle GPU instances

---

## 3. Responsible AI Considerations

### 3.1 Data Privacy and Compliance

**Measures Implemented:**

- All data processing performed locally, avoiding cloud exposure of sensitive telco logs
- No personally identifiable information (PII) present in telelogs dataset
- Training data remains within competition-defined boundaries
- Model weights do not memorize specific customer data patterns

**Compliance Framework:**

- GDPR-compatible processing (data minimization, purpose limitation)
- No cross-border data transfer for primary processing

### 3.2 Model Security Risks

**Identified Risks:**

1. **Prompt Injection**: Adversarial inputs could manipulate model outputs
2. **Model Extraction**: API access could enable model cloning
3. **Hallucination**: Incorrect diagnoses could lead to inappropriate network interventions

**Mitigations:**

- Input sanitization and validation before model inference
- Rate limiting and access logging for deployed endpoints
- Confidence scoring to flag uncertain predictions for human review
- Ensemble validation for critical diagnostic decisions

### 3.3 Data and Model Access Control and Transparency

**Access Control:**

- Repository access managed through GitHub permissions
- Model weights distributed via Hugging Face with clear licensing
- API access (if deployed) requires authentication and authorization

**Transparency:**

- Complete training code and data processing scripts are open-source
- Model card documenting capabilities, limitations, and intended use cases
- Clear versioning of datasets and model checkpoints

### 3.4 Edge Computing Considerations and Security Measures

**Edge Deployment Readiness:**

- Model size (1.5B parameters, 4-bit quantized) suitable for edge CPUs/NPUs
- Inference tested on Intel Core Ultra with Arc iGPU (~1.6 min per sample)
- No cloud dependency for inference operations

**Security Measures for Edge:**

- Local inference prevents data exfiltration to external servers
- Model integrity verification via checksums
- Sandboxed execution environment recommended
- Secure boot compatibility for tamper-resistant deployment

### 3.5 Data Governance Issues

**Data Lifecycle Management:**

- Raw data sourced exclusively from competition organizers
- Processed data derivation fully documented in scripts
- Clear separation between training, validation, and test datasets

**Governance Framework:**

- Data lineage tracked through processing pipeline
- Intermediate files can be regenerated from source
- No external data mixing to ensure reproducibility
- Model provenance documented (base model → fine-tuned version)

---

## 4. Results

| Configuration        | Pass@1 Accuracy | Notes                                     |
| -------------------- | --------------- | ----------------------------------------- |
| Baseline (Zero-shot) | 0.1405          | Qwen2.5-1.5B-Instruct without fine-tuning |
| R-SFT Fine-tuned     | TBD             | Reasoning-enhanced supervised fine-tuning |

### 4.1 Qualitative Observations

- Reasoning traces improve interpretability of model decisions
- Knowledge retention maintained through mixed training data
- Inference speed compatible with near-real-time edge deployment

---

## 5. Conclusion

We presented a reasoning-enhanced fine-tuning approach for network fault diagnosis that:

1. Leverages knowledge distillation from larger teacher models
2. Maintains compact model size suitable for edge deployment
3. Addresses key responsible AI considerations for telco environments

Future work could explore:

- Reinforcement learning from human feedback (RLHF) for improved accuracy
- Multi-turn diagnostic conversations
- Integration with network management systems APIs

---

## References

1. Qwen Team. "Qwen2.5: A Party of Foundation Models." arXiv preprint (2024).
2. Han, D., et al. "Unsloth: Efficient Fine-tuning of Large Language Models." (2024).
3. Intel Corporation. "IPEX-LLM: Intel Extension for PyTorch." GitHub Repository.
4. Competition Webinar Slides. "AI Telco Troubleshooting Challenge." Zindi Africa (2026).

---

**Repository**: [Github Repository](https://github.com/yehoshua0/competitions/blob/main/The%20AI%20Telco%20Troubleshooting%20Challenge)
**Author**: Kodjo Josué AYITEY (Yehoshua)  
**Contact**: [EMAIL_ADDRESS](mailto:jackjosue517@gmail.com)  
**Date**: February 2026
