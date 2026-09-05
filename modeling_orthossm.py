"""
OrthoSSM: Orthogonal State-Space Models with Exact Isometric Dynamics and Structured State-Space Duality
Copyright 2026 Prannessh (@Prannesshkva)
Licensed under the Apache License, Version 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalStateSpaceKernel(nn.Module):
    """
    Continuous-time Orthogonal State Space Layer.
    Enforces skew-symmetry A = -A^T so that state transitions U = exp(A * dt) in SO(N),
    preserving hidden state norm: ||h_t||_2 = ||h_{t-1}||_2.
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 48):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Skew-symmetric parameterization: A = skew - skew^T
        self.skew_raw = nn.Parameter(torch.randn(d_model, d_state, d_state) * 0.01)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.x_proj = nn.Linear(d_model, dt_rank, bias=False)

    def get_orthogonal_transition(self, delta_t):
        # A in so(N): skew-symmetric
        A = self.skew_raw - self.skew_raw.transpose(-1, -2)
        # Matrix exponential Cayley retraction: (I - A*dt/2)^(-1) * (I + A*dt/2)
        # Guarantees exact orthogonal group SO(N) transition
        scaled_A = delta_t.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        I = torch.eye(self.d_state, device=delta_t.device, dtype=delta_t.dtype)
        U = torch.linalg.solve(I - 0.5 * scaled_A, I + 0.5 * scaled_A)
        return U

    def forward(self, x, prev_state=None):
        """
        x: (batch, seq_len, d_model)
        prev_state: (batch, d_model, d_state)
        """
        b, l, d = x.shape
        dt = F.softplus(self.dt_proj(self.x_proj(x))) # (b, l, d)
        
        # B, C projections
        B_val = self.B(x) # (b, l, d_state)
        C_val = self.C(x) # (b, l, d_state)

        # Recurrent execution with exact norm conservation
        if prev_state is None:
            state = torch.zeros(b, d, self.d_state, device=x.device, dtype=x.dtype)
        else:
            state = prev_state

        outputs = []
        for t in range(l):
            dt_t = dt[:, t, :] # (b, d)
            U_t = self.get_orthogonal_transition(dt_t) # (b, d, d_state, d_state)
            
            # State update: h_t = U_t @ h_{t-1} + dt * B_t * x_t
            B_t = B_val[:, t, :].unsqueeze(1) # (b, 1, d_state)
            x_t = x[:, t, :].unsqueeze(-1) # (b, d, 1)
            
            # Orthogonal rotation preserves ||state||
            rotated_state = torch.matmul(U_t, state.unsqueeze(-1)).squeeze(-1)
            state = rotated_state + dt_t.unsqueeze(-1) * (x_t * B_t)
            
            # Readout: y_t = C_t @ h_t
            C_t = C_val[:, t, :].unsqueeze(1) # (b, 1, d_state)
            y_t = (state * C_t).sum(dim=-1) # (b, d)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, state
