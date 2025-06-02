import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', dropout=None):
        super(FinalAttentionQKV, self).__init__()
        
        self.attention_input_dim = attention_input_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_type = attention_type

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_out = nn.Linear(attention_hidden_dim, 1)
        self.b_in = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1, ))
        self.rate = nn.Parameter(torch.ones(1, ))
        
        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        batch_size, time_step, _ = input.size()
        input_q = self.W_q(torch.mean(input, dim=1)) # b h
        input_k = self.W_k(input)# b t h
        input_v = self.W_v(input)# b t h
        zeta_original = 0
        decay_term = 0

        if self.attention_type == 'add': # B * T * I  @ H * I
            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim)) # B * 1 * H
            h = q + input_k + self.b_in # b t h
            h = self.tanh(h) # B * T * H
            e = self.W_out(h).squeeze() # b t
        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1)) # B * h * 1
            dot_product = torch.matmul(input_k, q).squeeze() # b t
            time_miss = torch.log(1 + (1 - self.sigmoid(dot_product)))
            zeta_original = dot_product
            # TOFIX:
            decay_term = self.rate * time_miss 
            e = dot_product - decay_term
        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1, time_step, 1) # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1) # B * T * 2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba # B * T * 1
            e = torch.reshape(e, (batch_size, time_step)) # b t 
        
        a = self.softmax(e) # B * T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze() # B * I

        return v, a, zeta_original, decay_term


class AnchCare(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=32, output_dim=1, dropout=0.0, *args, **kwargs):
        super(AnchCare, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        num_classes = 2
        self.num_prototypes = 10 * num_classes
        self.prototype_vectors = nn.Parameter(torch.rand((self.num_prototypes, hidden_dim)))
        
        self.Q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.num_prototypes, 1)
        self.output = nn.Linear(self.num_prototypes, hidden_dim)
        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul', dropout=dropout)

        self.GRUs = nn.ModuleList([
            nn.GRU(1, self.hidden_dim, batch_first=True) for _ in range(self.input_dim)
        ])
        for i in range(32, 45):
            for p in self.GRUs[i].parameters():
                p.requires_grad=False
        
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        batch_size, _, feature_dim = input.size()
        lens = mask.sum(dim=1)

        GRU_embeded_input = self.GRUs[0](pack_padded_sequence(input[:, :, 0].unsqueeze(-1), lens.cpu(), batch_first=True, enforce_sorted=False))[1].squeeze().unsqueeze(1) # b 1 h
        for i in range(feature_dim - 1):
            embeded_input = \
            self.GRUs[i + 1](pack_padded_sequence(input[:, :, i + 1].unsqueeze(-1), lens.cpu(), batch_first=True, enforce_sorted=False))[1].squeeze().unsqueeze(1) # b 1 h
            GRU_embeded_input = torch.cat((GRU_embeded_input, embeded_input), 1)

        posi_input = self.dropout(GRU_embeded_input)  # batch_size * d_input * hidden_dim
        posi_input, attn, zeta_original, decay_term = self.FinalAttentionQKV(posi_input)
        distance = self.attention_similarity(posi_input, batch_size)  # b, p, d
        output = self.output(distance)
        return output
        
    def attention_similarity(self, x: torch.Tensor, batch_size: int):
        x1 = x.unsqueeze(1).repeat(1, self.num_prototypes, 1)  # b, p, d
        x2 = self.prototype_vectors.unsqueeze(0).repeat(batch_size, 1, 1)  # b, p, d

        q = self.relu(self.Q(x1))  # b, p, d
        k = self.relu(self.K(x2).transpose(-1, -2))  # b, d, p
        distance = torch.matmul(q, k)  # b, p, p
        distance = self.dropout(distance)
        distance = self.out(distance).squeeze(-1)  # b, p
        return distance