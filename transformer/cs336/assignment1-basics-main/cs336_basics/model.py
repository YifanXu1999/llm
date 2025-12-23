import torch
import torch.nn as nn
import einops
class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, device, dtype):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(out_dim, in_dim, device=device, dtype=dtype))
    
    def __init_weights__(self):
        mean = 0.0
        variance = 2 / (self.in_dim + self.out_dim)
        std = math.sqrt(variance)
        self.weights = nn.init.trunc_normal_(self.weights, mean=mean, std=std, a=-3 * std, b=3 * std)
    
    def forward(self, in_features):
        # in_features.shape = (batch_size, len, in_dim)
        # self.weights.shape = (out_dim, in_dim)
        # result = (batch_size, len, out_dim)
        result = einops.einsum(in_features, self.weights, 
        'batch len in_dim, out_dim in_dim -> batch len out_dim')
        return result


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device, dtype):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
    
    def __init_weights__(self):
        mean = 0.0
        std = 1.0
        self.embeddings = nn.init.trunc_normal_(self.weights, mean=mean, std=1, a=-3, b=3)
    
    def forward(self, token_ids):
        # token_ids.shape = (batch_size, len)
        # self.weights.shape = (vocab_size, d_model)
        # result = (batch_size, len, d_model)
        return  self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps,
        device,
        dtype,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.__init_weights__()
    
    def __init_weights__(self):
        self.weight = nn.init.ones_(self.weight)
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        a_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms_a = torch.sqrt(a_square + self.eps)  # (batch_size, len, d_model)
        rms_x = x / rms_a
        result = rms_x * self.weight
        return result.to(in_dtype)