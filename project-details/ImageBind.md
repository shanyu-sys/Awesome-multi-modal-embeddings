### ImageBind: One Embedding Space To Bind Them All

#### Paper

* [Link](https://arxiv.org/pdf/2305.05665.pdf)
* Conference: CVPR (highlighted paper)
* Year: 2023
* #cited: 8

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
