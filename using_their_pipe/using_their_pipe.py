from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

from shared import *

model_id = "CompVis/stable-diffusion-v1-4"

def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token=True, 
    )
    pipe = pipe.to(DEVICE)

    num_images = 3
    # prompt = "a photo of an astronaut riding a horse on mars"
    prompt = "a photo of an astronaut doing a handstand on a horse"
    # with autocast(DEVICE_STR):
    images = pipe(
        [prompt] * num_images, 
        guidance_scale=7.5, # 7 ~ 8.5
        num_inference_steps=5, 
    ).images

    grid = image_grid(images, rows=1, cols=num_images)
        
    grid.save(prompt.replace(' ', '_') + '.png')

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

main()
