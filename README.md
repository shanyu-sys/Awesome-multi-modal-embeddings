# Awesome-multi-modal-embeddings

## More than 3 modalities

### ImageBind: One Embedding Space To Bind Them All

#### Paper

* [Link]()
* Conference: CVPR (highlighted paper)
* Year: 2023
* #cited: 

* **modalities**: images, text, audio, depth, thermal, and IMU data

#### Code

* [Github](https://github.com/facebookresearch/ImageBind)
* Stars: **6.7k**
* license: Creative Commons Public Licenses

#### Installation and get embeddings

##### Installation: 

`pip install .`

* dep: torch 1.13+ ; python 3.8

##### get embeddings:

```python
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

def compute_image_embedding(image_paths):
  inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),}
  with torch.no_grad():
    embeddings = model(inputs)
  return embeddings[ModalityType.VISION]


def compute_text_embedding(text_list):
  inputs = {ModalityType.TEXT: data.load_and_transform_text(text_list, device),}
  with torch.no_grad():
    embeddings = model(inputs)
  return embeddings[ModalityType.TEXT]

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

### 

## vision-language

### LAVIS - A Library for Language-Vision Intelligence

#### Code

* [Github](https://github.com/salesforce/LAVIS/tree/main)
* Stars: **6k**
* License: BSD-3-Clause license

#### paper

A library with multiple papers and models including

* **InstructBLIP**: [preprint](https://arxiv.org/abs/2305.06500)
* **BLIP-2**: [ICML 2023](https://arxiv.org/abs/2301.12597)
* **Img2LLM-VQA**: [CVPR 2023](https://arxiv.org/pdf/2212.10846.pdf)

#### Installation and get embeddings

##### Installation: 

```
pip install salesforce-lavis
```

##### get embeddings: 

[demo](https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb)

```python
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

def get_image_text_embeddings(image, caption):
  model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
	image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
	text_input = txt_processors["eval"](caption)
	sample = {"image": image, "text_input": [text_input]}
	features_image = model.extract_features(sample, mode="image")
	features_text = model.extract_features(sample, mode="text")
	print(features_image.image_embeds.shape)
	# torch.Size([1, 32, 768])
	print(features_text.text_embeds.shape)
	# torch.Size([1, 12, 768])
  return featuers_image, features_text

raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")
caption = "a large fountain spewing water into the air"
```

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









#### Paper

* [Link]()
* Conference: 
* Year: 2023
* tasks

#### Code

* [Github]()
* Stars: **k**
* license: 

#### Installation and get embeddings

##### Installation: 

`pip install .`

* dep: torch 1.13+ ; python 3.8

##### get embeddings:

```python
```

