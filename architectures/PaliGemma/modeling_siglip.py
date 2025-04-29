from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbedding(nn.Module): 
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.num_patches = (config.image_size//config.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=config.patch_size, stride=config.patch_size, padding="valid")
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids", 
            torch.arange(self.num_patches).expand((1,-1)), 
            persistent=False
        )
    def forward(self, pixel_values: torch.FloatTensor)->torch.Tensor: 
        # pixel_values : [batch_size, channels, height, width]
        x = self.patch_embedding(pixel_values) # [batch_size, embed_dim, num_patches**-0.5, num_patches**-0.5]
        # flatten
        x = x.flatten(2) # [batch_Size, embed_dim, num_patches]
        x = x.permute(0,2,1) # [batch_size, num_patches, embed_dim]
        # add positional embedding
        x = x + self.pos_embed(self.position_ids) # [batch_size, num_patches, embed_dim]
        return x 

class SiglipAttention(nn.Module): 
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = (self.embed_dim//self.num_heads) 
        self.scale = (self.head_dim) ** -0.5
        self.dropout = self.config.attention_dropout

        self.k_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.v_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.q_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.out_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)

    def forward(self, x:torch.Tensor)-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x : [batch_size, num_patches, embed_dim] 
        
        ## 1 : Create query , key and value
        query = self.q_proj(x) # [batch_size, num_patches, embed_dim]
        key = self.k_proj(x)   # [batch_size, num_patches, embed_dim]
        value = self.v_proj(x) # [batch_Size, num_patches, embed_dim]

        ## 2 : break embed_dum into num_heads and head_dim ---> embed_dim = num_heads * head_dim
        query = query.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)     # [batch_size, num_patches, num_head, head_dim]
        key = key.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)         # [batch_size, num_patches, num_head, head_dim]
        value = value.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)     # [batch_size, num_patches, num_head, head_dim]

        ## 3. transpose the dimension
        query = query.transpose(1,2)    # [batch_size, num_heads, num_patches, embed_dim]
        key = key.transpose(1,2)        # [batch_size, num_heads, num_patches, embed_dim]
        value = value.transpose(1,2)    # [batch_size, num_heads, num_patches, embed_dim]

        ## 4. Dot product  = (Q.K^T)/root(d_model)
        attn_weights = (query @ key.transpose(2,3) * self.scale)    # [batch_size, num_heads, num_patches, num_patches]
        
        ## 5. apply softmax row-wise
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)     # [batch_size, num_heads, num_patches, num_patches]

        ## apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p =self.dropout, training=self.training)         # [batch_size, num_heads, num_patches, num_patches]

        ## 6. Mutiply attn_weights with value (weighted sum)
        attn_output = attn_weights @ value      # [batch_size, num_heads, num_patches, num_patches] @ [batch_size, num_heads, num_patches, head_size] --> [batch_size, num_heads, num_patches, head_size]

        ## 7. transpose the attn_output 
        attn_output = attn_output.transpose(1,2).contiguous()   # [batch_size, num_patches, num_heads, head_size]

        ## 8. Concat the attn_output: num_heads * head_size = embed_dim
        attn_output = attn_output.reshape(x.shape[0], x.shape[1], self.embed_dim)   # [batch_size, num_patches, embed_dim]

        ## 9. applying out_proj on attn_output
        attn_output = self.out_proj(attn_output)  # [batch_size, num_patches, embed_dim]

        return attn_output, attn_weights
class SiglipMLP(nn.Module): 
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__()
        self.config = config
        embed_dim = self.config.hidden_size

        self.fc1 = nn.Linear(in_features=embed_dim, out_features=self.config.intermediate_size)
        self.fc2 = nn.Linear(in_features=self.config.intermediate_size, out_features=embed_dim)
    
    def forward(self, x:torch.Tensor)->torch.Tensor: 
        # x : [batch_size, num_patches, embed_dim]
        x = self.fc1(x)                                     # [batch_size, num_patches, embed_dim]
        x = nn.functional.gelu(x, approximate="tanh")       # [batch_size, num_patches, embed_dim]
        x = self.fc2(x)                                     # [batch_size, num_patches, embed_dim]
        return x 
    
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps  =self.config.layer_norm_eps)
        self.self_attn = SiglipAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps = self.config.layer_norm_eps)
        self.mlp = SiglipMLP(self.config)

    def forward(self, hidden_states: torch.Tensor)-> torch.Tensor: 
        # hidden_states : [batch_size, num_pathces, embed_dim]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)     # [batch_size, num_pathces, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states)       # [batch_size, num_pathces, embed_dim]
        hidden_states = hidden_states + residual            # [batch_size, num_pathces, embed_dim]
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)     # [batch_size, num_pathces, embed_dim]
        hidden_states = self.mlp(hidden_states)             # [batch_size, num_pathces, embed_dim]
        hidden_states = hidden_states + residual            # [batch_size, num_pathces, embed_dim]
        return hidden_states


class SiglipEncoder(nn.Module): 
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
    def forward(self, x: torch.Tensor)-> torch.Tensor: 
        # x: [batch_size, num_pathces, embed_dim]
        for layer in self.layers: 
            x = layer(x)
        return x # [batch_size, num_pathces, embed_dim]  --> contexualized embedding
 
class SiglipVisionTransformer(nn.Module): 
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    def forward(self, pixel_values: torch.Tensor)-> torch.Tensor: 
        # pixel_values: [batch, channels, height, width]
        x = self.embeddings(pixel_values)  # [batch_size, num_pathces, embed_dim]
        x = self.encoder(x)    # [batch_size, num_patches, embed_dim]
        x = self.post_layernorm(x)
        return x 


class SiglipVisionModel(nn.Module): 
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple: 
        # [batch_size, channels, height, width] --> [batch_size, num_pathces, embed_dim]
        return self.vision_model(pixel_values)