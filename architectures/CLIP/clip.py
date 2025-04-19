from collections import OrderedDict
import os
import numpy as np
from dataclasses import dataclass

import torch
from torchvision import transforms as T
from torch import nn
import urllib

from simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm
from typing import Union, List


_tokenizer = _Tokenizer()

@dataclass
class ModelArgs:
    embed_dim: int = 512
    image_resolution: int = 234
    vision_layers: int = 12
    vision_width: int = 768
    vision_patch_size: int = 32
    context_length: int = 77
    vocab_size: int = 49408
    transformer_width: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 12

class LayerNorm(nn.LayerNorm): 
    # LayerNorm to handle fp16
    def forward(self, x: torch.Tensor): 
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QuickGELU(nn.Module): 
    def forward(self, x:torch.Tensor): 
        return x * torch.sigmoid(1.702*x) 
    
class ResidualAttentionBlock(nn.Module): 
    def __init__(self, d_model: int, n_head:int, attn_mask: torch.Tensor= None): 
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model*4)), 
            ("gelu", QuickGELU()), 
            ("c_proj", nn.Linear(4*d_model, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x:torch.Tensor): 
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x,x,x, need_weights = False, attn_mask = self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor): 
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module): 
    def __init__(self, width:int, layers:int, heads: int, attn_mask: torch.Tensor = None): 
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x:torch.Tensor): 
        return self.resblocks(x) # x: NLE
    
class VisionTransformer(nn.Module): 
    def __init__(self, input_resolution: int, patch_size: int, width:int, layers:int, heads:int, output_dim:int): 
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias = False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale*torch.randn(width))
        self.positional_embedding = nn.Parameter(scale* torch.randn((input_resolution//patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale*torch.randn(width, output_dim))

    def forward(self,x:torch.Tensor): 
        x = self.conv1(x) # [batch_size, width, grid, gird]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch_size, width, grid**2]
        x = x.permute(0, 2, 1) # [batch_size, grid**2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # [batch_size, grid**2 +1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
    
        x = x.permute(1, 0, 2)# NLE --> LNE (reference : input of shape [L,N,E] when `batch_first`= False in `nn.MultiheadAttention`)
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LNE --> NLE
        x = self.ln_post(x[:,0,:]) # [batch_size, 1, width]
        if self.proj is not None:
            x = x @ self.proj

        return x
    
class CLIP(nn.Module): 
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width//64 # head_size = 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution, 
            patch_size = vision_patch_size, 
            width=vision_width, 
            layers = vision_layers, 
            heads=vision_heads, 
            output_dim=embed_dim
        )
        self.transformer = Transformer(
            width = transformer_width, 
            layers = transformer_layers, 
            heads = transformer_heads, 
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self): 
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1) # zero out the lower diagonal
        return mask 
    
    @property
    def dtype(self): 
        return self.visual.conv1.weight.dtype
    
    def encode_image(self,image): 
        return self.visual(image.type(self.dtype))
    
    def encode_text(self, text): 
        # text : [batch_size, context_length]
        x = self.token_embedding(text).type(self.dtype) # [batch_size, context_length, d_model]
        x = x + self.positional_embedding.type(self.dtype) # [batch_size, context_length, d_model]

        x = x.permute(1,0,2) # NLD --> LND
        x = self.transformer(x) 
        x = x.permute(1,0,2) # LND --> NLD
        x = self.ln_final(x).type(self.dtype)

        # x : [batch_size, context_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def forward(self, image, text): 
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features /= image_features.norm(dim=1, keepdim = True)
        text_features /= text_features.norm(dim=1, keepdim = True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(model_path:str, device: str = "cpu"):
    checkpoint = torch.jit.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model = CLIP(**ModelArgs().__dict__)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


def get_preprocessor(n_px):
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    return T.Compose([
        T.Resize(n_px), 
        T.CenterCrop(n_px),
        _convert_image_to_rgb,
        T.ToTensor(), 
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result



def load():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"ViT-B-32.pt")
    url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

    if os.path.isfile(model_path): 
        model = build_model(model_path)
        preprocessor = get_preprocessor(model.visual.input_resolution)
    else  :
        with urllib.request.urlopen(url) as source, open(model_path, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))
        print("Model downloaded successfully")
        model = build_model(model_path)
        preprocessor = get_preprocessor(model.visual.input_resolution)

    return model, preprocessor
    

if __name__ == "__main__": 
    model, preprocessor = load()
    print("successful")