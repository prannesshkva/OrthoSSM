# OrthoSSM: Orthogonal State-Space Models with Exact Isometric Dynamics and Structured State-Space Duality

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22177116.svg)](https://doi.org/10.5281/zenodo.22177116)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-OrthoSSM%20Collection-ffd21e.svg)](https://huggingface.co/Prannesshkva)

Official repository for **OrthoSSM**, an advanced continuous-time sequence modeling architecture featuring exact orthogonal norm-preserving dynamics, skew-symmetric Lie-algebraic generators, and Structured State-Space Duality (SSD).

---

## Model Zoo & Hugging Face Checkpoints

| Model | Parameters | Architecture | Weights & Checkpoints |
| :--- | :---: | :---: | :--- |
| **OrthoSSM-130M** | 130M | Pure Continuous-Time Orthogonal SSM | [Prannesshkva/OrthoSSM-130M](https://huggingface.co/Prannesshkva/OrthoSSM-130M) |
| **OrthoSSM-Mamba-Falcon-Hybrid** | 195M | Interleaved Orthogonal Mamba + Falcon Attention | [Prannesshkva/OrthoSSM-Mamba-Falcon-Hybrid](https://huggingface.co/Prannesshkva/OrthoSSM-Mamba-Falcon-Hybrid) |
| **OrthoSSM-Qwen2.5-1.5B-Instruct** | 1.5B | Qwen2.5 Hybrid Instruct Model | [Prannesshkva/OrthoSSM-Qwen2.5-1.5B-Instruct](https://huggingface.co/Prannesshkva/OrthoSSM-Qwen2.5-1.5B-Instruct) |
| **OrthoSSM-Falcon-40B** | 40B | High-Capacity Foundation Hybrid | [Prannesshkva/OrthoSSM-Falcon-40B](https://huggingface.co/Prannesshkva/OrthoSSM-Falcon-40B) |

---

## Interactive Live Benchmarks
- **SSM Benchmark Suite**: [Prannesshkva/OrthoSSM-Benchmark](https://huggingface.co/spaces/Prannesshkva/OrthoSSM-Benchmark)
- **Samba Hybrid Benchmark**: [Prannesshkva/OrthoSSM-Samba-Benchmark](https://huggingface.co/spaces/Prannesshkva/OrthoSSM-Samba-Benchmark)

---

## Mathematical Architecture

OrthoSSM formalizes continuous-time recurrence with exact energy conservation:

$$\frac{dh(t)}{dt} = A h(t) + B x(t), \quad y(t) = C h(t) + D x(t)$$

where the generator $A$ is strictly constrained to the Lie algebra $\mathfrak{so}(N)$:
$$A = -A^T$$

The discretized state transition matrix:
$$U = \exp(A \Delta t) \in \mathrm{SO}(N)$$
is an exact orthogonal matrix, guaranteeing:
$$\|h_t\|_2 = \|h_{t-1}\|_2$$

This guarantees zero gradient explosion and zero vanishing memory over infinite sequence contexts.

---

## Quickstart

```bash
pip install torch transformers huggingface_hub
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Prannesshkva/OrthoSSM-130M"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)

inputs = tokenizer("Orthogonal state-space models enable", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Citation & Zenodo DOI

```bibtex
@software{prannessh_orthossm_2026,
  author       = {Prannessh},
  title        = {OrthoSSM: Orthogonal State-Space Models with Exact Isometric Dynamics and Structured State-Space Duality},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22177116},
  url          = {https://doi.org/10.5281/zenodo.22177116}
}
```

---

## License & Attribution

- Released under Apache License 2.0.
- Novel continuous-time orthogonal dynamics and hybrid state-space duality contributions by Prannessh (@Prannesshkva).
- See `NOTICE` for upstream derivative notices and acknowledgments.
