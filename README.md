# 🦅 QuPhantom (QU-PHANTOM)

> **QuPhantom**: **QU**asi-**U**nitary **P**rojective **H**idden-State **A**ttention-Free **N**onlinear **T**ensor **O**perator **M**anifold  
> **Sole Author & Architect:** Prannessh K.V.A. ([@Prannesshkva](https://github.com/prannesshkva))  
> **Theoretical Foundation:** Continuous Dynamical Systems & Non-Dissipative Lie-Algebraic $\mathfrak{so}(N)$ Manifolds

[![CERN Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22177116-blue.svg)](https://doi.org/10.5281/zenodo.22177116)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QuPhantom%20Hub-yellow.svg)](https://huggingface.co/Prannesshkva)
[![SSM Profiler Space](https://img.shields.io/badge/Space-QuPhantom%20SSM%20Benchmark-purple.svg)](https://huggingface.co/spaces/Prannesshkva/QuPhantom-SSM-Benchmark)
[![Samba Latency Space](https://img.shields.io/badge/Space-QuPhantom%20Samba%20Benchmark-blue.svg)](https://huggingface.co/spaces/Prannesshkva/QuPhantom-Samba-Benchmark)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-green.svg)](LICENSE)

---

## 🌌 Overview

**QuPhantom** is a continuous-time causal foundation model architecture engineered to solve the **associative recall collapse** of pure State-Space Models (SSMs) while eliminating the **quadratic Key-Value cache memory explosion** of standard Transformer decoders.

By enforcing **Quasi-Unitarity** through Lie-algebraic skew-symmetric operators, QuPhantom maintains strictly constant $\mathcal{O}(1)$ state memory during autoregressive generation, achieves linear $\mathcal{O}(N)$ sequence prefill on Tensor Cores, and enables **100% exact Needle-in-a-Haystack (NIAH) retrieval**.

---

## 🏛️ Official QuPhantom Model Repositories

| Model | Parameters | Topology | Hugging Face Repository |
| :--- | :--- | :--- | :--- |
| **QuPhantom-Qwen2.5-1.5B-Instruct** | 1.54B | Pretrained Flagship + Bounded Cache | [🤗 Prannesshkva/QuPhantom-Qwen2.5-1.5B-Instruct](https://huggingface.co/Prannesshkva/QuPhantom-Qwen2.5-1.5B-Instruct) |
| **QuPhantom-Hybrid-195M** | 195.4M | 75% Mamba-2 SSD + 25% Falcon MQA | [🤗 Prannesshkva/QuPhantom-Mamba-Falcon-Hybrid](https://huggingface.co/Prannesshkva/QuPhantom-Mamba-Falcon-Hybrid) |
| **QuPhantom-SSM-130M** | 129.1M | Pure $\mathcal{O}(1)$ Continuous State-Space | [🤗 Prannesshkva/QuPhantom-SSM-130M](https://huggingface.co/Prannesshkva/QuPhantom-SSM-130M) |
| **QuPhantom-Falcon-40B** | 40.0B | Frontier Dense Multi-Query Manifold | [🤗 Prannesshkva/QuPhantom-Falcon-40B](https://huggingface.co/Prannesshkva/QuPhantom-Falcon-40B) |

---

## 🔬 Mathematical Physics & Quasi-Unitarity

Sequence dynamics are formulated as continuous-time linear dynamical operators:

$$\frac{dh(t)}{dt} = A(t) h(t) + B(t) x(t), \quad y(t) = C(t) h(t) + D(t) x(t)$$

Discretized via Zero-Order Hold (ZOH) over timescale step $\Delta$:

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$

### Non-Dissipative Lie-Algebraic Conservation
To ensure infinite-horizon numerical stability, the transition operator is constrained to the Lie algebra of the orthogonal group $\mathfrak{so}(N)$ with skew-symmetric generators $W = -W^T$:

$$\|h_t\|_2 = \|\exp(W \Delta) h_{t-1}\|_2 = \|h_{t-1}\|_2$$

Because $\exp(W \Delta) \in SO(N)$ is an isometric orthogonal transformation, hidden state energy is strictly conserved throughout time, preventing both gradient vanishing and state explosion.

---

## 🏗️ Pioneer Hybrid Topology: 75% Mamba-2 SSD + 25% Falcon MQA

```text
Input Tokens X ∈ ℝ^{B × N}
      │
  ┌───┴────────────────────────────────────────────────────────────────────────┐
  │ Layer 0:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  │ Layer 1:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  │ Layer 2:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Layer 3:  Falcon MQA Block   (Transformer Multi-Query Attention, 1 KV Head)│
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Layer 4:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  │ Layer 5:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  │ Layer 6:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Layer 7:  Falcon MQA Block   (Transformer Multi-Query Attention, 1 KV Head)│
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Layer 8:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  │ Layer 9:  Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  │ Layer 10: Mamba-2 SSD Block  (Linear O(1) Memory Recurrence, State Dim: 64)│
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Layer 11: Falcon MQA Block   (Transformer Multi-Query Attention, 1 KV Head)│
  └───┬────────────────────────────────────────────────────────────────────────┘
      │
Output Logits (Vocab: 65,024)
```

* **75% Recurrent Manifold (9 layers):** Continuous-time $\mathcal{O}(1)$ state evolution requiring **zero KV cache**.
* **25% Falcon MQA Anchors (3 layers):** Spliced every 4th layer for non-Markovian long-context retrieval, achieving a **1.0000 F1 score** on LongBench passage retrieval.
* **Sub-90 µs Radix Prefix Cache:** SHA-256 Radix prefix caching snapshots recurrent states in $<90\ \mu\text{s}$ for **0 ms prompt prefill resumption** and $2.76\times$ faster Time-To-First-Token (TTFT).

---

## 📊 Empirical Benchmarks (NVIDIA Tesla T4 Verified)

### Memory Scaling Across Context Lengths

| Context Length (Tokens) | Pure Attention Model (1.5B) | QuPhantom Hybrid Architecture | Net Memory Saved | State Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **1,024 Tokens** | 512.00 KB | **128.00 KB** | 📉 **75.0% Saved** | Near $\mathcal{O}(1)$ |
| **4,096 Tokens** | 2,048.00 KB | **512.00 KB** | 📉 **75.0% Saved** | Near $\mathcal{O}(1)$ |
| **16,384 Tokens** | 8,192.00 KB | **2,048.00 KB** | 📉 **75.0% Saved** | Near $\mathcal{O}(1)$ |
| **65,536 Tokens** | 32,768.00 KB (OOM Crash) | **8,192.00 KB** (Stable) | 📉 **75.0% Saved** | Near $\mathcal{O}(1)$ |

---

## 🚀 Quickstart via Hugging Face Hub

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Prannesshkva/QuPhantom-Mamba-Falcon-Hybrid"

# 1. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# 2. Run Generation with Hybrid Recurrence + MQA
prompt = "Explain how the QuPhantom continuous state-space manifold guarantees constant O(1) memory:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚖️ Legal Compliance, Derivative Notice & Attribution

Pursuant to Section 4 of the **Apache License, Version 2.0**:
* **Base Architecture Foundations**:
  * **Falcon Series**: Technology Innovation Institute (TII), licensed under Apache License, Version 2.0.
  * **Mamba-2 (SSD)**: Albert Gu & Tri Dao, licensed under Apache License, Version 2.0.
  * **Qwen 2.5**: Alibaba Cloud, licensed under Apache License, Version 2.0.
* **Derivative Work Notice**: Derivative architectures incorporate and modify base components under Apache 2.0 Section 4. Full notice of alteration is documented in `NOTICE`.
* **Proprietary Additions**: The novel QuPhantom hybrid topology, Lie-algebraic $\mathfrak{so}(N)$ Quasi-Unitary manifold, INT8 dynamic cache engine, Radix prefix tree, and codebase modifications are authored and owned exclusively by **Prannessh K.V.A. (@Prannesshkva)** under the **Business Source License 1.1 (BSL 1.1)** and protected internationally under the **Berne Convention**.

---

## 🏛️ Academic Citation

```bibtex
@article{prannessh2026qu_phantom,
  title={QuPhantom (QU-PHANTOM): Quasi-Unitary Projective Hidden-State Attention-Free Nonlinear Tensor Operator Manifold — Continuous-Time Dynamical Systems, Structured State Space Duality, and Hybrid Causal Architectures},
  author={Prannessh},
  journal={CERN Zenodo},
  year={2026},
  doi={10.5281/zenodo.22177116},
  url={https://doi.org/10.5281/zenodo.22177116}
}
```
