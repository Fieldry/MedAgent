from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class RETAINLayer(nn.Module):
    """RETAIN layer.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    This layer is used in the RETAIN model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: the hidden feature size.
        dropout: dropout rate. Default is 0.5.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.beta_gru = nn.GRU(input_dim, input_dim, batch_first=True)

        self.alpha_li = nn.Linear(input_dim, 1)
        self.beta_li = nn.Linear(input_dim, input_dim)

    @staticmethod
    def reverse_x(input, lengths):
        """Reverses the input."""
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input

    def compute_alpha(self, rx, lengths):
        """Computes alpha attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        attn_alpha = torch.softmax(self.alpha_li(g), dim=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        """Computes beta attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        attn_beta = torch.tanh(self.beta_li(h))
        return attn_beta

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            c: a tensor of shape [batch size, input_dim] representing the
                context vector.
        """
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        rx = self.reverse_x(x, lengths)
        attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)
        a = attn_alpha * attn_beta # (patient, sequence len, input_dim)
        c = a * x  # (patient, sequence len, input_dim)
        c = torch.sum(c, dim=1)  # (patient, input_dim)
        return c, a


class RETAIN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, **kwargs):
        super(RETAIN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.retain_layer = RETAINLayer(input_dim, dropout)
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None):
        out, attn = self.retain_layer(x, mask)
        out = self.proj(out)
        return out, attn


if __name__ == "__main__":
    x = torch.randn(2, 13, 75)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    model = RETAIN(75, 128)
    out, attn = model(x, mask)
    print(out.shape)
    print(attn.shape)