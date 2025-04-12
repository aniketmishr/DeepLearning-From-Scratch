import numpy as np
from PIL import Image
import torch
from model import VisionTransformer

k = 10
config = {
    'img_size':384,
    'patch_size':16,
    "in_chans":3,
    "n_classes":1000,
    "embed_dim":768,
    "depth":12,
    "n_heads":12,
    "mlp_ratio":4.,
    "qkv_bias":True,
    'p':0.,
    "attn_p":0.,
}

imagenet_labels = dict(enumerate(open("classes.txt")))

checkpoint = torch.load("weight/model.pth", weights_only=False)
model = VisionTransformer(**config)
model.load_state_dict(checkpoint, strict=True)
model.eval()

pil_img = Image.open(f"assets/{input('Enter image Name: ')}").resize((384,384))
assert len(pil_img.getbands()) == 3, "The number of channels of Image must be 3"
img = (np.array(pil_img) / 128) - 1  # in the range -1, 1
inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")