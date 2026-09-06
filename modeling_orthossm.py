"""
OrthoSSM: Orthogonal State-Space Models with Exact Isometric Dynamics and Structured State-Space Duality
Copyright 2026 Prannessh (@Prannesshkva)
Licensed under the Apache License, Version 2.0

Key Architectural Capabilities:
1. Exact Lossless Invertibility: U^(-1) = U^T enables O(1) state rollback for tree search (MCTS).
2. Continuous-Time Physical Dynamics: Accepts continuous irregular timestamps dt in R^+.
3. Block-Orthogonal Subspace Memory: Non-interfering Lie algebra frequency blocks eliminate attention dilution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class OrthogonalStateSpaceKernel(nn.Module):
    """
    Continuous-Time Orthogonal State-Space Layer with Exact Lie-Algebraic Isometry.
    
    Mathematical Foundations:
    - Continuous system: dh(t)/dt = A h(t) + B x(t)
    - Lie algebra constraint: A in so(N) => A = -A^T (skew-symmetric)
    - Cayley retraction: U = (I - 0.5 * dt * A)^(-1) * (I + 0.5 * dt * A) in SO(N)
    - Exact Unitarity/Isometry: U^T U = I, ||h_t||_2 = ||h_{t-1}||_2
    - Exact Invertibility: h_{t-1} = U_t^T (h_t - dt_t * (x_t * B_t))
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 48):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Skew-symmetric parameterization: A = skew - skew^T in so(d_state)
        self.skew_raw = nn.Parameter(torch.randn(d_model, d_state, d_state) * 0.02)
        
        # Input & output projections
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Time-scale projections
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.x_proj = nn.Linear(d_model, dt_rank, bias=False)

    def get_skew_matrix(self) -> torch.Tensor:
        """Returns skew-symmetric generator A in so(N), where A = -A^T."""
        return self.skew_raw - self.skew_raw.transpose(-1, -2)

    def get_orthogonal_transition(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Computes exact orthogonal transition matrix U in SO(d_state) via Cayley retraction.
        delta_t: (batch, d_model)
        Returns: U of shape (batch, d_model, d_state, d_state)
        """
        A = self.get_skew_matrix()  # (d_model, d_state, d_state)
        # Scaled generator: (batch, d_model, d_state, d_state)
        scaled_A = delta_t.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0)
        
        # Identity matrix: (1, 1, d_state, d_state)
        I = torch.eye(self.d_state, device=delta_t.device, dtype=delta_t.dtype).view(1, 1, self.d_state, self.d_state)
        
        # Cayley retraction: (I - 0.5 * dt * A)^(-1) * (I + 0.5 * dt * A)
        U = torch.linalg.solve(I - 0.5 * scaled_A, I + 0.5 * scaled_A)
        return U

    def step_forward(
        self, 
        x_t: torch.Tensor, 
        prev_state: torch.Tensor, 
        dt_phys: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step recurrent execution forward in time.
        
        Args:
            x_t: (batch, d_model) input at time t
            prev_state: (batch, d_model, d_state) state at time t-1
            dt_phys: continuous physical time multiplier (default 1.0)
            
        Returns:
            y_t: (batch, d_model) output
            next_state: (batch, d_model, d_state) state at time t
            dt_t: (batch, d_model) effective dt used
        """
        # Compute dt: (batch, d_model)
        dt_learned = F.softplus(self.dt_proj(self.x_proj(x_t)))
        dt_t = dt_learned * dt_phys
        
        # Orthogonal transition matrix: (batch, d_model, d_state, d_state)
        U_t = self.get_orthogonal_transition(dt_t)
        
        # B projection: (batch, d_state) -> (batch, 1, d_state)
        B_t = self.B(x_t).unsqueeze(1)
        x_in = x_t.unsqueeze(-1)  # (batch, d_model, 1)
        
        # Orthogonal rotation: preserves norm ||prev_state||
        rotated_state = torch.matmul(U_t, prev_state.unsqueeze(-1)).squeeze(-1)
        
        # State update: h_t = U_t @ h_{t-1} + dt_t * (x_t * B_t)
        next_state = rotated_state + dt_t.unsqueeze(-1) * (x_in * B_t)
        
        # Readout: y_t = C_t @ h_t + D * x_t
        C_t = self.C(x_t).unsqueeze(1)  # (batch, 1, d_state)
        y_t = (next_state * C_t).sum(dim=-1) + self.D * x_t
        
        return y_t, next_state, dt_t

    def step_backward(
        self, 
        x_t: torch.Tensor, 
        current_state: torch.Tensor, 
        dt_phys: float = 1.0
    ) -> torch.Tensor:
        """
        EXACT LOSSLESS REVERSE STEP (Rewind / Rollback):
        Reconstructs previous state h_{t-1} from current state h_t using U^(-1) = U^T.
        
        No matrix inversion or KV-cache cloning required!
        
        Args:
            x_t: (batch, d_model) input at time t
            current_state: (batch, d_model, d_state) state at time t
            dt_phys: physical time multiplier matching the forward step
            
        Returns:
            prev_state: (batch, d_model, d_state) exact state at time t-1
        """
        dt_learned = F.softplus(self.dt_proj(self.x_proj(x_t)))
        dt_t = dt_learned * dt_phys
        
        # Orthogonal transition U_t: (batch, d_model, d_state, d_state)
        U_t = self.get_orthogonal_transition(dt_t)
        
        # Transpose U_t^T == U_t^(-1)
        U_inv = U_t.transpose(-1, -2)
        
        # Subtract input contribution
        B_t = self.B(x_t).unsqueeze(1)
        x_in = x_t.unsqueeze(-1)
        unrotated_state = current_state - dt_t.unsqueeze(-1) * (x_in * B_t)
        
        # Exact reverse rotation: h_{t-1} = U_t^T @ unrotated_state
        prev_state = torch.matmul(U_inv, unrotated_state.unsqueeze(-1)).squeeze(-1)
        return prev_state

    def rollback(
        self, 
        trajectory_x: List[torch.Tensor], 
        final_state: torch.Tensor,
        trajectory_dt_phys: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Multi-step exact state rollback across an entire trajectory.
        Rewinds final_state back to the initial state without caching intermediate states.
        
        Args:
            trajectory_x: list of x_t tensors in forward chronological order [x_1, x_2, ..., x_K]
            final_state: state at step K
            trajectory_dt_phys: optional list of physical dt values
            
        Returns:
            initial_state: exact reconstructed state at step 0
        """
        state = final_state
        K = len(trajectory_x)
        for i in reversed(range(K)):
            dt_phys = trajectory_dt_phys[i] if trajectory_dt_phys is not None else 1.0
            state = self.step_backward(trajectory_x[i], state, dt_phys=dt_phys)
        return state

    def forward(
        self, 
        x: torch.Tensor, 
        prev_state: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence forward pass supporting both regular and continuous irregular sampling.
        
        Args:
            x: (batch, seq_len, d_model)
            prev_state: optional initial state (batch, d_model, d_state)
            timestamps: optional continuous physical timestamps (batch, seq_len) in seconds.
                        If provided, dt_phys_t = timestamps[t] - timestamps[t-1].
        """
        b, l, d = x.shape
        device = x.device
        dtype = x.dtype
        
        if prev_state is None:
            state = torch.zeros(b, d, self.d_state, device=device, dtype=dtype)
        else:
            state = prev_state

        # Precompute dt_phys if continuous timestamps provided
        if timestamps is not None:
            # dt_phys[:, 0] = timestamps[:, 0], dt_phys[:, t] = timestamps[:, t] - timestamps[:, t-1]
            dt_phys = torch.zeros_like(timestamps)
            dt_phys[:, 0] = F.relu(timestamps[:, 0]) + 1e-4
            dt_phys[:, 1:] = F.relu(timestamps[:, 1:] - timestamps[:, :-1]) + 1e-4
        else:
            dt_phys = None

        outputs = []
        for t in range(l):
            x_t = x[:, t, :]
            dt_p = dt_phys[:, t].unsqueeze(-1) if dt_phys is not None else 1.0
            y_t, state, _ = self.step_forward(x_t, state, dt_phys=dt_p)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, state


class BlockOrthogonalStateSpaceKernel(nn.Module):
    """
    Multi-Frequency Block-Orthogonal State-Space Kernel.
    
    Partitions the state space into K mutually orthogonal subspaces:
    h = [h_1, h_2, ..., h_K] where each h_k in R^(d_k).
    
    Because the generator A is strictly block-diagonal:
        A = blkdiag(A_1, A_2, ..., A_K), with A_k in so(d_k),
    subspaces rotate independently at distinct frequencies omega_k:
    - High-frequency blocks: track fast-changing local token transitions
    - Low-frequency blocks: preserve long-range invariants indefinitely
    - Mutual non-interference: Subspace i cannot leak into Subspace j,
      completely eliminating Attention Entropy Dilution.
    """
    def __init__(
        self, 
        d_model: int, 
        num_blocks: int = 4, 
        block_size: int = 8, 
        dt_rank: int = 48
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.total_state_dim = num_blocks * block_size
        self.dt_rank = dt_rank

        # Distinct skew blocks: (num_blocks, d_model, block_size, block_size)
        self.skew_blocks = nn.Parameter(
            torch.randn(num_blocks, d_model, block_size, block_size) * 0.02
        )
        
        # Frequency scaling factors per block (log-spaced from slow to fast)
        freq_init = torch.exp(torch.linspace(-2.0, 2.0, num_blocks)).view(num_blocks, 1, 1, 1)
        self.register_buffer("block_freqs", freq_init)

        self.B = nn.Linear(d_model, self.total_state_dim, bias=False)
        self.C = nn.Linear(d_model, self.total_state_dim, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.x_proj = nn.Linear(d_model, dt_rank, bias=False)

    def get_block_orthogonal_transitions(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Computes orthogonal transition matrices for each subspace block.
        delta_t: (batch, d_model)
        Returns: U of shape (num_blocks, batch, d_model, block_size, block_size)
        """
        # Skew-symmetry per block: A_k = -A_k^T
        A = self.skew_blocks - self.skew_blocks.transpose(-1, -2)
        A = A * self.block_freqs  # Apply multi-frequency scaling
        
        # (num_blocks, batch, d_model, block_size, block_size)
        dt_expanded = delta_t.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        scaled_A = dt_expanded * A.unsqueeze(1)
        
        I = torch.eye(self.block_size, device=delta_t.device, dtype=delta_t.dtype).view(1, 1, 1, self.block_size, self.block_size)
        U = torch.linalg.solve(I - 0.5 * scaled_A, I + 0.5 * scaled_A)
        return U

    def forward(
        self, 
        x: torch.Tensor, 
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with non-interfering block-orthogonal memory.
        x: (batch, seq_len, d_model)
        prev_state: (batch, d_model, num_blocks, block_size)
        """
        b, l, d = x.shape
        device = x.device
        dtype = x.dtype
        
        if prev_state is None:
            state = torch.zeros(b, d, self.num_blocks, self.block_size, device=device, dtype=dtype)
        else:
            state = prev_state

        dt_learned = F.softplus(self.dt_proj(self.x_proj(x)))  # (b, l, d)
        B_val = self.B(x).view(b, l, self.num_blocks, self.block_size)
        C_val = self.C(x).view(b, l, self.num_blocks, self.block_size)

        outputs = []
        for t in range(l):
            dt_t = dt_learned[:, t, :]  # (b, d)
            # U: (num_blocks, b, d, block_size, block_size)
            U_blocks = self.get_block_orthogonal_transitions(dt_t)
            
            x_t = x[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (b, d, 1, 1)
            B_t = B_val[:, t, :].unsqueeze(1)  # (b, 1, num_blocks, block_size)
            
            # Subspace-wise rotation: each block rotates in its own invariant subspace
            # state: (b, d, num_blocks, block_size) -> permute to (num_blocks, b, d, block_size, 1)
            st_perm = state.permute(2, 0, 1, 3).unsqueeze(-1)
            rot_perm = torch.matmul(U_blocks, st_perm).squeeze(-1)
            rotated_state = rot_perm.permute(1, 2, 0, 3)  # (b, d, num_blocks, block_size)
            
            # State update
            state = rotated_state + dt_t.unsqueeze(-1).unsqueeze(-1) * (x_t * B_t)
            
            # Readout across all orthogonal subspaces
            C_t = C_val[:, t, :].unsqueeze(1)  # (b, 1, num_blocks, block_size)
            y_t = (state * C_t).sum(dim=(-1, -2)) + self.D * x[:, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, state
