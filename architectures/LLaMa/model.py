import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
from dataclasses import dataclass
from typing import Optional 

@dataclass
class ModelArgs: 
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int]=None # number of heads for K and V (GQA)
    vocab_size: int = -1 # will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    # needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None


def precompute_theta_pos_frequencies(head_dim:int, seq_len:int, device:str, theta: float = 10000.0):
    assert head_dim%2==0, "Dimension must be divisible by 2"
    # build the theta parameters 
    # according to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,...,dim/2]
    # shape: (head_dim/2)
    theta_numerator = torch.arange(0,head_dim,2).float()
    # shape: (head_dim/2)
    theta = 1.0/(theta**(theta_numerator/head_dim)).to(device)
    # construct the positions (the 'm' parameter) 
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using the outer product 
    # shape: (seq_len) outer product (head_dim/2) --> (seq_ln, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # we can compute complex numbers in the polar form c = R* exp(i*m*theta), where R = 1 as follows: 
    # (seq_len, head_dim/2) --> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x:torch.Tensor, freqs_complex: torch.Tensor, device): 
    # (B,seq_len, H, head_dim) --> (B,seq_len,H,head_dim/2)  grouping consecutive pairs of x and converting to complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1,2))
    # (seq_len, head_dim/2) --> (1,seq_len,1,head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B,seq_len, H,head_dim/2) * (1, seq_len,1 , head_dim/2) * (B,seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B,seq_len, H, head_dim/2) --> (B,seq_len, H, head_dim/2,2)
    x_out = torch.view_as_real(x_rotated)
    # (B,seq_len, H, head_dim/2) --> (B,seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class RMSNorm(nn.Module): 
    def __init__(self, dim:int, eps:int = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # the gamma parameter
        self.eps = eps
    def _norm(self, x:torch.Tensor): 
        # (B,seq_len,dim)
        # rsqrt: 1/sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    def forward(self, x:torch.Tensor): 
        # (B,seq_len, dim)
        # (dim) * (B,seq_len,dim) --> (B,seq_len,dim)
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module): 
    def __init__(self,args:ModelArgs):
        super().__init__()
        '''
        1. GQA: Grouped Query Attention
        2. K-V cache
        '''
        self.n_heads_q = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim//self.n_heads_q
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # w_q, w_k, w_v matrices initialized according to GQA
        self.wq = nn.Linear(args.dim, self.n_heads_q*self.head_dim, bias = False) # (Dim->H_Q * H_Dim) : (Dim -> Dim)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False) # (Dim->H_KV * H_Dim) 
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False) # (Dim->H_KV * H_Dim) 
        # output matrix
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False) 

        # KV Cache 
        self.cache_k = torch.zero((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cahce_v = torch.zero((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex: torch.Tensor): 
        # (B,seq_len, Dim) --> (B,1,Dim) 
        batch_size, seq_len, _ = x.shape
        # (B,1,Dim) --> (B,1,H_Q * H_Dim)
        xq = self.wq(x)
        # (B,1,Dim) --> (B,1, H_KV * H_Dim)
        xk = self.wk(x)
        # (B,1,Dim) --> (B,1, H_KV * H_Dim)
        xv = self.wv(x)

        # (B,1, H_Q*H_Dim) --> (B,1,H_Q,H_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B,1, H_KV*H_Dim) --> (B,1,H_KV, H_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B,1,H_KV*H_Dim) --> (B,1,H_KV,H_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

## Apply roatary positional embedding to Q and K 
        xq = apply_rotary_embeddings(xq, freqs_complex,device=x.device) # Shape remains same
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device) # Shape remains same

        ## replace the entry in cache 
        self.cache_k[:batch_size,start_pos: start_pos+seq_len]  = xk
        self.cahce_v[:batch_size, start_pos:start_pos+seq_len] = xv

        ## retrieve key and values from cache 
        # (B, seq_len_KV, H_KV, H_Dim)
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        # (B, seq_len_KV, H_KV, H_Dim)
        values = self.cache_v[:batch_size, :start_pos+seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        # (B, seq_len_KV, H_KV, H_Dim) --> (B,seq_len_KV, H_Q, H_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, seq_len_KV, H_KV, H_Dim) --> (B,seq_len_KV, H_Q, H_Dim)
        values = repeat_kv(values, self.n_rep)

        
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

class FeedForward(nn.Module): 
    def __init__(self, args:ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2*hidden_dim/3)
        if args.ffn_dim_multiplier is not None: 
            hidden_dim = int(args.ffn_dim_multiplier*hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias = False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)

    def forward(self,x: torch.Tensor): 
        # (B,seq_len,dim) --> (B,seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # (B,seq_len, dim) --> (B,seq_len, hidden_dim)
        x_V = self.w3(x)
        # (B,seq_len,hidden_dim) * (B,seq_len,hidden_dim) --> (B,seq_len,hidden_dim)
        x = swish * x_V
        # (B,seq_len,hidden_dim) --> (B,seq_len,dim)
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module): 
    def __init__(self, args:ModelArgs): 
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_size = args.dim//args.n_heads
        # RMS Normalization BEFORE self attention 
        self.attention_norm = RMSNorm(self.dim, args.norm_eps)
        # self attention block
        self.attention = SelfAttention(args)
        # RMS Normalization BEFORE feed forward
        self.ffn_norm = RMSNorm(self.dim, args.norm_eps)
        # Feed forward layer
        self.feed_forward = FeedForward(args)
    def forward(self, x:torch.Tensor, start_pos:int ,freqs_complex: torch.Tensor): 
        # (B,seq_len,dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)  
        h = h + self.feed_forward.forward(self.ffn_norm(h))
        return h

class Transformer(nn.Module): 
    def __init__(self, args: ModelArgs): 
        super().__init__()
        assert args.vocab_size!=-1, "Vocab size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddigs = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers): 
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(in_features=args.dim, out_features=args.vocab_size, bias = False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len*2, device = self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int): 
        # (B, seq_len) -> (B,1)
        batch_size, seq_len = tokens.shape
        assert seq_len==1, "Only one token at a time can be processed"
        # (B,seq_len) --> (B.seq_len,dim)
        h = self.tok_embeddigs(tokens)

        # retrieve the pairs (m,theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        # consecutively apply all the encoder layers
        for layer in self.layers: 
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output