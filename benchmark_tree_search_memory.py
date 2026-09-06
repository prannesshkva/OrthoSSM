"""
Benchmark: Tree-Search Reasoning Memory Footprint (MCTS / o1-style Test-Time Search)
Comparing:
1. Standard Transformer KV-Cache Branching
2. OrthoSSM Reversible State Rollback (U^T)
"""

import math

def calculate_mcts_memory(
    n_layers: int = 32,
    hidden_size: int = 4096,
    num_heads: int = 32,
    head_dim: int = 128,
    branching_factor: int = 4,
    depth: int = 6,
    tokens_per_step: int = 64,
    d_state: int = 16,
    dtype_bytes: int = 2  # float16
):
    print("=" * 80)
    print("      MCTS REASONING TREE SEARCH MEMORY FOOTPRINT BENCHMARK")
    print("=" * 80)
    print(f"Model Configuration: Layers={n_layers}, Hidden={hidden_size}, Heads={num_heads}, HeadDim={head_dim}")
    print(f"Tree Search Parameters: Branching Factor={branching_factor}, Depth={depth}, Tokens/Step={tokens_per_step}")
    
    total_nodes = sum(branching_factor ** d for d in range(1, depth + 1))
    print(f"Total Trajectory Nodes Explored in Tree: {total_nodes:,}")
    print("-" * 80)

    # 1. Transformer KV Cache Memory
    # For every node at depth d, sequence length is d * tokens_per_step
    # KV cache per token: 2 * n_layers * hidden_size * dtype_bytes
    bytes_per_token_kv = 2 * n_layers * hidden_size * dtype_bytes
    
    total_transformer_bytes = 0
    for d in range(1, depth + 1):
        nodes_at_depth = branching_factor ** d
        seq_len_at_depth = d * tokens_per_step
        kv_bytes_at_node = seq_len_at_depth * bytes_per_token_kv
        total_transformer_bytes += nodes_at_depth * kv_bytes_at_node

    transformer_mb = total_transformer_bytes / (1024 ** 2)
    transformer_gb = total_transformer_bytes / (1024 ** 3)

    # 2. OrthoSSM Reversible State Rollback Memory
    # In OrthoSSM, the search only maintains:
    # - The single active state h: (hidden_size * d_state) * dtype_bytes
    # - The token IDs along the current branch: max_seq_len * 4 bytes (int32)
    # When exploring alternative branches, OrthoSSM simply rolls back state via U^T!
    # No intermediate node caches need to be kept.
    ortho_active_state_bytes = hidden_size * d_state * dtype_bytes
    max_tokens = depth * tokens_per_step
    token_history_bytes = max_tokens * 4  # int32 token IDs
    total_ortho_bytes = ortho_active_state_bytes + token_history_bytes
    
    ortho_kb = total_ortho_bytes / 1024
    ortho_mb = total_ortho_bytes / (1024 ** 2)

    print(f"{'Metric':<35} | {'Standard Transformer':<20} | {'OrthoSSM (Reversible)'}")
    print("-" * 80)
    print(f"{'Intermediate State Retention':<35} | {'Full KV Cache per Node':<20} | {'Zero (Rewound via U^T)'}")
    print(f"{'State Vector Size (per layer)':<35} | {f'{max_tokens} Tokens KV':<20} | {f'{d_state} Floats (Fixed)'}")
    print(f"{'Peak Search Memory (Depth 6)':<35} | {f'{transformer_gb:.2f} GB':<20} | {f'{ortho_kb:.2f} KB'}")
    
    reduction_factor = total_transformer_bytes / total_ortho_bytes
    print(f"{'Memory Reduction Factor':<35} | {'1x (Baseline)':<20} | {f'{reduction_factor:,.0f}x Less VRAM'}")
    print("=" * 80)

if __name__ == "__main__":
    calculate_mcts_memory()
