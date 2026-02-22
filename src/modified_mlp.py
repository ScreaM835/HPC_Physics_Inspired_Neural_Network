"""
Modified MLP with Trainable Random Fourier Features (RFF) for PINNs.

Implements:
  1. Trainable RFF input embedding (Wang, Yu & Perdikaris 2023)
     γ(x) = [cos(Bx), sin(Bx)]  where B is a trainable matrix.

  2. Modified MLP architecture (Wang, Teng & Perdikaris 2021)
     Skip connections inject the transformed input into every hidden
     layer via element-wise gating, preventing "coordinate forgetting."

References
----------
- Wang et al., "An Expert's Guide to Training Physics-Informed Neural
  Networks," arXiv:2308.08468 (2023).
- Wang, Teng & Perdikaris, "Understanding and Mitigating Gradient Flow
  Pathologies in Physics-Informed Neural Networks," SIAM Review (2021).
"""

from __future__ import annotations

import math
from typing import List

import deepxde as dde
import torch
import torch.nn as nn

# DeepXDE's PyTorch NN base class — provides the interface that
# dde.Model expects (auxiliary_vars, output_transform, etc.).
from deepxde.nn.pytorch.nn import NN as DdeNN


class RandomFourierFeatures(nn.Module):
    """Trainable Random Fourier Feature embedding layer.

    Maps a D-dimensional input x to a 2*num_features-dimensional vector:
        γ(x) = [cos(Bx + b), sin(Bx + b)]

    The frequency matrix B is initialised from N(0, sigma) and is
    *trainable* so the network can tune the frequencies to match the
    physics (e.g., to lock onto the QNM frequency ω ≈ 0.37).

    Parameters
    ----------
    in_features : int
        Dimension of the raw input (2 for [x*, t]).
    num_features : int
        Number of Fourier frequencies.  Output dimension = 2 * num_features.
    sigma : float
        Standard deviation of the initial frequency matrix B.
        Controls the initial frequency "spread".  For problems with
        characteristic frequencies ~O(1), sigma ∈ [1, 10] works well.
    trainable : bool
        If True (default), B participates in backpropagation.
    """

    def __init__(
        self,
        in_features: int = 2,
        num_features: int = 64,
        sigma: float = 1.0,
        trainable: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_features = num_features
        self.out_features = 2 * num_features

        # Frequency matrix B ~ N(0, sigma)
        B = torch.randn(in_features, num_features) * sigma
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, in_features)

        Returns
        -------
        Tensor (N, 2 * num_features)
        """
        # x @ B has shape (N, num_features)
        # Cast B to match input dtype (e.g. float64 training)
        B = self.B.to(dtype=x.dtype)
        projection = x @ B
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)


class ModifiedMLP(DdeNN):
    """Modified MLP with RFF input embedding and Wang-style skip connections.

    Inherits from DeepXDE's NN base class so it can be plugged directly
    into dde.Model (provides auxiliary_vars, output_transform, etc.).

    Architecture
    ------------
    1. RFF embedding:  x ∈ R^2  →  γ(x) ∈ R^{2*num_rff}
    2. Encoder U:      H_U = σ(W_U · γ + b_U)
    3. Encoder V:      H_V = σ(W_V · γ + b_V)
    4. Hidden layers (Wang modified):
         Z_k = σ(W_k · H_{k-1} + b_k)
         H_k = (1 - Z_k) ⊙ H_U  +  Z_k ⊙ H_V
    5. Output layer:   y = W_out · H_last + b_out

    The skip-connected encoders U and V inject the transformed input
    into every hidden layer, preventing deep layers from "forgetting"
    the original (x*, t) coordinates.

    Parameters
    ----------
    hidden_layers : list of int
        Width of each hidden layer, e.g. [80, 40, 20, 10].
    num_rff : int
        Number of Fourier frequencies (output dim = 2*num_rff).
    rff_sigma : float
        Initial std of the RFF frequency matrix.
    rff_trainable : bool
        Whether the RFF frequencies are trainable.
    activation : str
        Activation function name ('tanh', 'relu', 'gelu', 'sin').
    """

    _activations = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "gelu": nn.functional.gelu,
        "sin": torch.sin,
    }

    def __init__(
        self,
        hidden_layers: List[int],
        num_rff: int = 64,
        rff_sigma: float = 1.0,
        rff_trainable: bool = True,
        activation: str = "tanh",
    ):
        super().__init__()

        if activation not in self._activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {list(self._activations.keys())}"
            )
        self.activation_fn = self._activations[activation]

        # --- RFF embedding ---
        self.rff = RandomFourierFeatures(
            in_features=2,  # (x*, t)
            num_features=num_rff,
            sigma=rff_sigma,
            trainable=rff_trainable,
        )
        rff_out_dim = self.rff.out_features  # 2 * num_rff

        # --- Encoder layers U and V ---
        # These project the RFF features into the hidden layer width.
        # According to Wang et al. 2021, U and V are computed exactly once
        # and shared across all hidden layers to inject a consistent coordinate embedding.
        # This requires all hidden layers to have the same width.
        self._hidden_widths = list(hidden_layers)
        if len(set(self._hidden_widths)) > 1:
            raise ValueError("ModifiedMLP requires all hidden layers to have the same width.")
        
        width = hidden_layers[0]
        self.encoder_U = nn.Linear(rff_out_dim, width)
        self.encoder_V = nn.Linear(rff_out_dim, width)

        # --- Hidden layers ---
        self._hidden = nn.ModuleList()
        in_dim = rff_out_dim  # first hidden layer takes the RFF output directly
        for width in hidden_layers:
            self._hidden.append(nn.Linear(in_dim, width))
            in_dim = width

        # --- Output layer ---
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

        # --- Initialisation (Glorot uniform) ---
        self._init_weights()

    def _init_weights(self):
        """Glorot uniform initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, 2)  — columns are [x*, t]

        Returns
        -------
        Tensor (N, 1)  — predicted wave amplitude phi(x*, t)
        """
        # Apply DeepXDE input transform if set
        if self._input_transform is not None:
            x = self._input_transform(x)

        # Step 1: RFF embedding
        gamma = self.rff(x)  # (N, 2*num_rff)

        # Step 2-4: Hidden layers with Wang skip connections
        # Compute U and V embeddings once
        U = self.activation_fn(self.encoder_U(gamma))
        V = self.activation_fn(self.encoder_V(gamma))

        H = gamma
        for i, layer in enumerate(self._hidden):
            Z = self.activation_fn(layer(H))
            H = (1 - Z) * U + Z * V

        # Step 5: Output layer
        y = self.output_layer(H)

        # Apply DeepXDE output transform if set
        if self._output_transform is not None:
            y = self._output_transform(x, y)

        return y
