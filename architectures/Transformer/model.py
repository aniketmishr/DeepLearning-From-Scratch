import torch 
import torch.nn as nn 
import math

class InputEmbedding(nn.Module): 
    def __init__(self, d_model:int, vocab_size: int): 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x shape : (Batch, seq) 
        # return : (Batch, seq, dim)
        return self.embedding(x) * (self.d_model ** 0.5)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps = 1e-6): 
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # added
    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        return self.alpha* (x-mean)/(std+self.eps) + self.beta

class FeedForwardBlock(nn.Module): 
    def __init__(self, d_model: int, d_ff:int, dropout:float): 
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # (batch, seq_len, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # (batch, seq_len, d_model)
    def forward(self,x): 
        # x shape: (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadedAttentionBlock(nn.Module): 
    def __init__(self, d_model:int, h:int, dropout: int): 
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h ==0 , "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model,bias=False)
        self.w_k = nn.Linear(d_model,d_model,bias=False)
        self.w_v = nn.Linear(d_model,d_model,bias=False)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key,value, mask, dropout:nn.Dropout) :
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None: 
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value) , attention_scores
    
    def forward(self,q,k,v,mask): 
        # x shape : (batch , seq_len, d_model)
        # return : (batch , seq_len, d_model)

        query = self.w_q(q)     # (batch , seq_len, d_model)
        key = self.w_k(k)       # (batch , seq_len, d_model)
        value = self.w_v(v)     # (batch , seq_len, d_model)
        # (batch , seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1],self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1],self.h, self.d_k).transpose(1,2)

        x, attention_scores = MultiHeadedAttentionBlock.attention(query, key, value,mask, self.dropout)

        # Combine all the head together
        x = x.transpose(1,2).contiguous().flatten(2) # (batch, seq_len, h*d_k) --> (batch, seq_len, d_model)
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module): 
    def __init__(self, dropout: float): 
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self,x, sublayer): 
        out = x + self.dropout(sublayer(self.norm(x)))  # Skip connections
        return out

class EncoderBlock(nn.Module): 
    def __init__(self, self_attention_block: MultiHeadedAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float): 
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self,x, src_mask): 
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return self.dropout(x)
    
class Encoder(nn.Module): 
    def __init__(self, layers: nn.ModuleList): 
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask): 
        for layer in self.layers: 
            x = layer(x,mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module): 
    def __init__(self, self_attention_block: MultiHeadedAttentionBlock, cross_attention_block: MultiHeadedAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float): 
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, target_mask): 
        # x shape: (batch, seq_len, d_model)
        # encoder_output : (batch, seq_len, d_model)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x 
    
class Decoder(nn.Module): 
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, target_mask): 
        for layer in self.layers: 
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module): 
    def __init__(self, d_model, vocab_size): 
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x): 
        # x shape: (batch, seq_len, d_model)
        # output: (batch, seq_lem , vocab_size)
        return torch.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module): 
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask): 
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self,encoder_output, src_mask, tgt, target_mask ): 
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, target_mask)
    
    def project(self, x): 
        # (batch, seq_len, d_model)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model = 512, N=6, h= 8, dropout= 0.3, d_ff = 2048):

    # create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N): 
        encoder_self_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N): 
        decoder_self_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block , decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters(): 
        if p.dim() > 1: 
            nn.init.xavier_uniform_(p)
    
    return transformer

# if __name__ == "__main__":
    
#     model = build_transformer(src_vocab_size=100, tgt_vocab_size=100,src_seq_len=5, tgt_seq_len=5)
#     inp_seq = torch.Tensor([[1,2,3,4,5], [6,7,8,9,10]]).type(torch.long) # (Batch, seq_len) = (2,5)
#     e_o = model.encode(inp_seq , torch.ones(2,5))
#     print(e_o.shape)
#     d_o = model.decode(e_o, torch.ones(2,5), torch.ones(2,5))
#     print(d_o.shape)