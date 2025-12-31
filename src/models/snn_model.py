# src/models/snn_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from snntorch import surrogate



# Try surrogate from snntorch, else fallback
try:
    import snntorch as snn  # noqa
    from snntorch import surrogate
    spike_fn = surrogate.fast_sigmoid()
except Exception:
    class SimpleSurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            (inp,) = ctx.saved_tensors
            sigma = 10.0
            s = 1.0 / (1.0 + torch.exp(-sigma * inp))
            grad = sigma * s * (1 - s)
            return grad_output * grad

    def spike_fn(x):
        return SimpleSurrogate.apply(x)


# -----------------------------------------------------
# LIF Layer
# -----------------------------------------------------
class LIFLayer(nn.Module):
    """
    Clean and stable LIF layer with:
    - tau_mem leak
    - dropout on input currents
    - surrogate gradient spikes
    """

    def __init__(self, in_features, out_features, tau_mem=12.0, v_th=0.9, dropout=0.02):
        super().__init__()


        self.fc = nn.Linear(in_features, out_features)

        self.tau_mem = float(tau_mem)
        self.v_th = float(v_th)

        # Dropout for stability (recommended: 0.05)
        self.dropout = nn.Dropout(dropout)

        # Use fast-surrogate sigmoid for spikes
        self.spike_fn = spike_fn

    def forward(self, x_t, mem):
        """
        x_t: (batch, in_features)
        mem: (batch, out_features)
        """
        # Weighted input + dropout
        cur = self.fc(x_t)
        cur = self.dropout(cur)

        # Compute leak constant (Euler method)
        alpha = torch.exp(torch.tensor(-1.0 / self.tau_mem,
                                       device=x_t.device))

        # Update membrane potential
        mem = alpha * mem + cur

        # Spike using surrogate activation
        z = self.spike_fn(mem - self.v_th)

        # Reset after spike
        mem = mem - z * self.v_th

        return mem, z

# -----------------------------------------------------
# SNN Encoder
# -----------------------------------------------------
class SNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 128),
                 embedding_dim=64, tau_mem=12.0, dropout=0.0,
                 read_k_fraction=0.2):

        super().__init__()
        h1, h2 = hidden_dims
        self.h1 = h1
        self.h2 = h2

        self.lif1 = LIFLayer(in_dim, h1, tau_mem=tau_mem, dropout=dropout)
        self.lif2 = LIFLayer(h1, h2, tau_mem=tau_mem, dropout=dropout)

        self.readout = nn.Sequential(
            nn.Linear(h2, max(h2 // 2, embedding_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(max(h2 // 2, embedding_dim), embedding_dim)
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.read_k_fraction = float(read_k_fraction)

    def forward(self, spikes):
        # normalize shape
        if spikes.dim() != 3:
            raise ValueError("Spikes must be (B,T,D) or (T,B,D).")

        if spikes.shape[0] > spikes.shape[1]:
            spikes_t = spikes  # (T,B,D)
            T, B, _ = spikes_t.shape
        else:
            B, T, _ = spikes.shape
            spikes_t = spikes.permute(1, 0, 2)

        device = spikes_t.device

        mem1 = torch.zeros(B, self.h1, device=device)
        mem2 = torch.zeros(B, self.h2, device=device)

        mem2_hist = []

        for t in range(T):
            x_t = spikes_t[t].float()
            mem1, z1 = self.lif1(x_t, mem1)
            mem2, z2 = self.lif2(z1, mem2)
            mem2_hist.append(mem2)

        mem2_stack = torch.stack(mem2_hist, dim=0)

        k = max(1, int(self.read_k_fraction * T))
        last_k = mem2_stack[-k:].mean(dim=0)

        emb = self.readout(last_k)
        emb = self.norm(emb)
        emb = F.normalize(emb, p=2, dim=1)

        return emb
