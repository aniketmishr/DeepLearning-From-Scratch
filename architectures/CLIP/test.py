import clip
import torch 
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path

# load the model 
device  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"on device: {device}")
model, preprocess = clip.load()

# prepare the inputs 
images = [f"images/{f.name}" for f in Path("images").iterdir() if f.is_file()]

image_input = torch.stack([preprocess(Image.open(image)) for image in images]).to(device) # (n_imgs, 3, 224, 224)
text_inputs = clip.tokenize(input("Enter the prompt: ")).to(device) # (1,77)

# calculate features 
print("Starting model inference\n")
with torch.no_grad(): 
    image_features = model.encode_image(image_input) # (4, embed), embed = 512
    text_features = model.encode_text(text_inputs) # (1, embed), embed = 512


# pick the top 5 most familiar labels for the image 
image_features/= image_features.norm(dim=-1, keepdim=True)
text_features/= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)  # (1, embed) @ (embed, 4) --> (1,4)
values, indices = similarity[0].topk(1)
print(f"probability : {(values.item()*100):.3f}%")

## showing searched image

plt.imshow(Image.open(images[indices.item()]))
plt.show()