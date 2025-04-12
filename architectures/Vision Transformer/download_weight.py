import timm
import torch
from pathlib import Path
import os

model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

# Save the model weights
weight_path = Path('weight').mkdir(exist_ok=True)
torch.save(model_official.state_dict(), os.path.join(weight_path, "model.pth"))