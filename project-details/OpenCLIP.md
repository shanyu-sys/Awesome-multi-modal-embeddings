### OpenCLIP

An open source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training).

#### Paper

* [Link](https://arxiv.org/abs/2212.07143)
* Conference: preprint
* Year: 2022

#### Code

* [Github](https://github.com/mlfoundations/open_clip)
* Stars: **5.6k**
* license: 

#### Installation and get embeddings

##### Installation: 

`pip install open_clip_torch`

##### get embeddings:

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
def get_image_embedding(image):
  with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
	return image_features

text = tokenizer(["a diagram", "a dog", "a cat"])
def get_text_features(text):
  with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
  return text_features
```