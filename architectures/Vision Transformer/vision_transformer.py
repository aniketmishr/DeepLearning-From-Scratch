
import torch 
import torch.nn as nn

class PatchEmbedding(nn.Module): 
    '''
    Breaks the input image into patches, and converts the patches into embeddings
    '''
    def __init__(self, img_size, patch_size, in_chans=3, embd_dim = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.n_patches = (img_size//patch_size)**2

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embd_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor: 
        '''
        Shape of input tensor: (n_samples, in_chans, img_size, img_size)

        Shape of the output tensor: (n_samples, patch_size, embd_dim)
        '''
        # x.shape : (n_samples, in_chans, img_size, img_size)

        x = self.proj(x)       # shape: (n_samples, embd_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)       # shape: (n_samples, embd_dim, n_patches)
        x = x.transpose(1,2)   # shape: (n_samples, n_patches, embd_dim)
        return x
    

class Attention(nn.Module): 
    '''
    Communication Module: Allows tokens to communicate with each other and share context
    '''
    def __init__(self, dim, n_heads, attn_p = 0.2, proj_p = 0.2): 
        super().__init__()
        self.dim= dim
        self.n_heads = n_heads
        self.head_size = dim // n_heads
        self.scale = self.head_size**-0.5
        self.qkv = nn.Linear(dim, dim*3 ) # (Token) -> (n_heads * head_size * 3(one for each q,k,v))
        self.attn_p = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim,dim)
        self.proj_p = nn.Dropout(proj_p)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        '''
        Input Tensor Shape : (n_samples, n_patches+1, dim)
        Output Tensor Shape : (n_samples, n_patches+1, dim)
        '''
        n_samples, n_tokens, dim = x.shape
        if dim!=self.dim: 
            return ValueError
        qkv = self.qkv(x) # shape: (n_samples, n_patches+1, dim*3)
        qkv = qkv.reshape( 
            n_samples, n_tokens, 3, self.n_heads, self.head_size
        ) # shape : (n_samples, n_patches+1, 3, n_heads, head_size)

        qkv = qkv.permute(
            2,0,3,1,4
        ) # shape : (3, n_samples, n_heads, n_patches+1, head_size)

        q,k,v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2,-1) # (n_samples, n_heads, head_size, n_patches+1)

        dp = (q @ k_t) * self.scale # (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_p(attn)
        weighted_avg = attn @ v # (n_samples, n_heads, n_patches+1, head_size)
        ## No need for masking as we are creating encoder (ViT only uses Encoder)
        weighted_avg = weighted_avg.transpose(1,2) # (n_samples, n_patches+1, n_heads, head_size)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches+1, dim)
        x = self.proj(x) # (n_samples, n_patches+1, dim)
        x = self.proj_p(x)
        return x # (n_samples, n_patches+1, dim)

class Multilayer(nn.Module): 
    '''
    Computational Block : Where tokens are allowed to think independently
    '''
    def __init__(self, in_features, hidden_features,out_features, p = 0.2): 
        super().__init__()
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p)
    def forward(self,x: torch.Tensor)->torch.Tensor: 
        # x: (n_samples, n_patches+1, dim)
        x = self.fc1(x)                     #(n_samples, n_patches+1, dim)
        x = self.dropout(self.act(x))       #(n_samples, n_patches+1, dim)
        x  = self.fc2(x)                    #(n_samples, n_patches+1, dim)
        x = self.dropout(x)                 #(n_samples, n_patches+1, dim)
        return x

class Block(nn.Module): 
    '''
    Leverages both Communication and Computation: Tokens are allowed to communicate with each other
    and made to think, independently
    '''
    def __init__(self, dim, n_heads, mlp_ratio = 4.0,p=0.2, attn_p=0.2): 
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, 1e-6)
        self.norm2 = nn.LayerNorm(dim, 1e-6)
        self.attention = Attention(
            dim = dim, 
            n_heads=n_heads,
            attn_p=attn_p,
            proj_p=p
        )
        hidden_features = int(mlp_ratio*dim)
        self.mlp = Multilayer(
            in_features=dim, 
            hidden_features= hidden_features,
            out_features=dim,
            p =p
        )
    def forward(self,x: torch.Tensor)-> torch.Tensor: 
        '''
        Input tensor shape: (n_samples, n_patches+1, dim)
        Output tensor shape: (n_samples, n_patches+1, dim)
        '''
        x = x+ self.attention(self.norm1(x)) # skip connection
        x = x+ self.mlp(self.norm2(x))       # skip connection
        return x

class VisionTransformer(nn.Module): 
    '''
    Takes image an input and predicts the class label
    '''
    def __init__(self, img_size = 384, patch_size = 16,in_chans= 3, 
    n_classes = 1000, embed_dim = 768, depth = 12, n_heads = 12,
    mlp_ratio = 4.0, p = 0.2, attn_p= 0.2): 
        super().__init__()
        self.patch_embd = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embd_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.patch_embd.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim = embed_dim,
                    n_heads=n_heads, 
                    mlp_ratio=mlp_ratio, 
                    p= p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, 1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
    def forward(self, x: torch.Tensor)-> torch.Tensor: 
        '''
        x: (n_samples, chans, img_size, img_size)
        returns logits: 
        (n_samples, n_classes)
        '''
        n_samples = x.shape[0]
        x = self.patch_embd(x) # (n_samples, n_patches, dim)
        cls_token = self.cls_token.expand(
            n_samples, -1,-1
        ) # (n_samples, 1, dim)
        x = torch.cat([cls_token, x],dim=1)  # (n_samples,1 +n_patches, dim)
        for block in self.blocks:
            x = block(x) # (n_samples,1 +n_patches, dim)
        
        x = self.norm(x)
        final_cls_token = x[:,0]     # (n_samples, dim)
        x = self.head(final_cls_token) # (n_samples, n_clases)
        return x



if __name__ == "__main__":
    imgs = torch.randn(4, 3, 384,384)
    model = VisionTransformer()
    model.eval()
    with torch.no_grad():
        pred = model(imgs)
        print(pred.shape)
    # print(pred)
    print("Model working with Zero Errors :)")
