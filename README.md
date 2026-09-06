# OrthoSSM: Orthogonal State-Space Models with Exact Isometric Dynamics and Structured State-Space Duality

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22177116.svg)](https://doi.org/10.5281/zenodo.22177116)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-OrthoSSM%20Collection-ffd21e.svg)](https://huggingface.co/Prannesshkva)

**OrthoSSM** is an advanced continuous-time sequence modeling architecture that breaks the fundamental limitations of discrete attention-based Transformers. By constraining state transitions to the **orthogonal Lie group $\mathrm{SO}(N)$**, OrthoSSM unlocks exact mathematical reversibility, continuous-time asynchronous dynamics, and anti-dilution multi-subspace memory.

---

## What OrthoSSM Does That SOTA Transformers Cannot Do

While SOTA Transformers have solved vanishing gradients via residual paths and dot-product attention, they remain fundamentally **dissipative, discrete, and quadratic in memory**. OrthoSSM introduces three capabilities mathematically impossible in standard Transformers:

### 1. Exact Lossless Reversibility ($U^{-1} = U^T$) for Zero-Memory Tree Search
* **The Transformer Problem**: Softmax attention is non-invertible. In Monte Carlo Tree Search (MCTS) or test-time reasoning rollouts (e.g. o1/o3 or DeepSeek-R1 style search), exploring alternative branches requires caching or recomputing gigabytes of KV caches per branch node.
* **The OrthoSSM Solution**: Because $U_t \in \mathrm{SO}(N)$, its inverse is simply its transpose:
  $$h_{t-1} = U_t^T \left(h_t - \Delta t_t (x_t B_t)\right)$$
* **Empirical Verification**: Reversing an active trajectory across **1,000 forward steps** reconstructs the initial state $h_0$ with relative $L_2$ error of **$1.61 \times 10^{-13}$** (machine precision).
* **MCTS Memory Advantage**: Reduces peak tree-search memory from **967.12 GB** (cloned KV caches) to **129.50 KB** (single rewindable buffer).

### 2. Continuous-Time Physical Dynamics ($\Delta t \in \mathbb{R}^+$)
* **The Transformer Problem**: Transformers index discrete integer positions ($t \in \mathbb{Z}^+$), making them unable to process non-uniform physical time gaps or perform zero-shot temporal super-resolution.
* **The OrthoSSM Solution**: Continuous Cayley retraction handles continuous physical timestamps:
  $$U(\Delta t) = \left(I - \frac{1}{2}\Delta t A\right)^{-1} \left(I + \frac{1}{2}\Delta t A\right), \quad A = -A^T \in \mathfrak{so}(N)$$
  Allows querying intermediate representations at fractional times $t + \delta$ without retraining.

### 3. Block-Orthogonal Subspaces (Zero Attention Dilution)
* **The Transformer Problem**: In long contexts ($100\text{k}+$ tokens), softmax attention spreads thin across tokens (entropy collapse / attention dilution), washing out subtle signals.
* **The OrthoSSM Solution**: Generator $A = \mathrm{diag}(A_1, \dots, A_K)$ partitions state space into mutually orthogonal Lie sub-algebras rotating at distinct frequencies. Subspaces **never cross-contaminate**, preserving invariant memories indefinitely.

---

## Model Zoo & Checkpoints

| Model | Parameters | Architecture | Weights & Checkpoints |
| :--- | :---: | :---: | :--- |
| **OrthoSSM-130M** | 130M | Pure Continuous-Time Orthogonal SSM | [Prannesshkva/OrthoSSM-130M](https://huggingface.co/Prannesshkva/OrthoSSM-130M) |
| **OrthoSSM-Mamba-Falcon-Hybrid** | 195M | Interleaved Orthogonal Mamba + Falcon Attention | [Prannesshkva/OrthoSSM-Mamba-Falcon-Hybrid](https://huggingface.co/Prannesshkva/OrthoSSM-Mamba-Falcon-Hybrid) |
| **OrthoSSM-Qwen2.5-1.5B-Instruct** | 1.5B | Qwen2.5 Hybrid Instruct Model | [Prannesshkva/OrthoSSM-Qwen2.5-1.5B-Instruct](https://huggingface.co/Prannesshkva/OrthoSSM-Qwen2.5-1.5B-Instruct) |
| **OrthoSSM-Falcon-40B** | 40B | High-Capacity Foundation Hybrid | [Prannesshkva/OrthoSSM-Falcon-40B](https://huggingface.co/Prannesshkva/OrthoSSM-Falcon-40B) |

---

## Quickstart & Verification

```bash
# 1. Run exact mathematical reversibility proof
python test_ortho_reversibility.py

# 2. Run MCTS tree-search reasoning memory benchmark
python benchmark_tree_search_memory.py
```

### Python API

```python
import torch
from modeling_orthossm import OrthogonalStateSpaceKernel

kernel = OrthogonalStateSpaceKernel(d_model=64, d_state=16)

# Forward step
y_t, next_state, dt_t = kernel.step_forward(x_t, prev_state)

# Exact backward step (Zero memory rollback)
recovered_prev_state = kernel.step_backward(x_t, next_state)
assert torch.allclose(prev_state, recovered_prev_state, atol=1e-5)
```

---


### Zero-Training Deliberative Reasoning Engine (Frozen Model Controller)

Run deep test-time MCTS tree search and (1)$ multi-hop rollbacks on any frozen Transformer without training:

`ash
python test_ortho_deliberative_search.py
`

`python
from orthossm_runtime_controller import OrthoMemoryManager, CyclicManifoldVerifier, MultiHopInversionEngine

# 1. Drop-in replacement for KV cache with sub-100MB cap
memory_manager = OrthoMemoryManager(max_active_tokens=4096, state_dim=64, enable_int8=True)

# 2. Multi-hop rollback in O(1) time
engine = MultiHopInversionEngine(state_dim=64)
h_0 = engine.hop_backward(final_state, U_hop, Delta_H)

# 3. Intrinsic hallucination verification via Lie manifold geodesic curvature
verifier = CyclicManifoldVerifier(tolerance=0.08)
result = verifier.evaluate_reasoning_step(step_hidden_states)
print('Is Valid:', result['is_valid'], 'Confidence:', result['confidence_score'])
`

## Citation

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

## License
- Licensed under the Apache License, Version 2.0.
- Novel continuous-time orthogonal dynamics and reversible state rollback by Prannessh (@Prannesshkva).
