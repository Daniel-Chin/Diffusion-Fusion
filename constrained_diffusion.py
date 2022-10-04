'''
pixel-level constraints
    symmetry
        symmetric dog
    横看成岭侧成峰
        computer + music
'''

from contextlib import nullcontext

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from winsound import Beep
logging.set_verbosity_error()

from shared import *

PROMPT = 'a photo of a symmetric bottle on kitchen table'
WIDTH, HEIGHT = 512, 512
num_inference_steps = 150
guidance_scale = 7.5
N_EXP = 1
LATENT_FILE = 'latents_constrained_diff.tensor'

generator = torch.Generator(DEVICE_STR).manual_seed(2333)
tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14", 
)
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, 
    beta_schedule="scaled_linear", num_train_timesteps=1000, 
)

print('Loading models...')
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", 
).to(DEVICE)

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae", 
    use_auth_token=True, 
).to(DEVICE)

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", 
    use_auth_token=True, 
).to(DEVICE)
print('Loaded models.')

def forward():    
    print('Encoding text...')
    text_input = tokenizer(
        PROMPT, padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt", 
    )
    text_embeddings = text_encoder(
        text_input.input_ids.to(DEVICE), 
    )[0]

    uncond_input = tokenizer(
        [""], padding="max_length", 
        max_length=tokenizer.model_max_length, 
        return_tensors="pt", 
    )
    uncond_embeddings = text_encoder(
        uncond_input.input_ids.to(DEVICE)
    )[0]

    text_embeddings = torch.cat(
        [uncond_embeddings] * N_EXP + 
        [text_embeddings] * N_EXP, 
    )
    text_embeddings = torch.cat([text_embeddings] * N_EXP)

    print('Sampling latent...')
    latents = torch.randn((
        1, unet.in_channels, HEIGHT // 8, WIDTH // 8,
    ), generator=generator)
    latents = latents.to(DEVICE)
    latents = torch.cat([latents] * N_EXP)
    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.sigmas[0]

    with torch.autocast(DEVICE_STR) if (
        HAS_CUDA or torch.__version__ > '1.12.2'
    ) else nullcontext():
        for i, t in tqdm([*enumerate(scheduler.timesteps)], 'denoising'):
            print()
            latents = oneStep(
                i, t, latents, guidance_scale, 
                text_embeddings,  
            )

    print('Saving...')
    with open(LATENT_FILE, 'wb') as f:
        torch.save(latents, f)
    
def visualize():
    with open(LATENT_FILE, 'rb') as f:
        latents = torch.load(f)
    
    latents = 1 / 0.18215 * latents

    print('Decoding image...')
    images = toImg(vae.decode(latents).sample)

    print('plotting...')
    DPI = 300
    matplotlib.rcParams['figure.dpi'] = DPI
    matplotlib.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(
        WIDTH * 1.1 / DPI, 
        HEIGHT * 1.1 / DPI, 
    ))
    ax = fig.subplots(1, 1)
    ax.imshow(images[0])
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('out_con_dif.png', dpi=DPI)
    # plt.show()
    from console import console
    console({**globals(), **locals()})

def smartLineBreak(x, line_width=20):
    buffer = []
    words = x.split(' ')
    acc = 0
    for word in words:
        acc += len(word) + 1
        if acc > line_width:
            buffer.append('\n')
            acc = len(word) + 1
        buffer.append(word)
        buffer.append(' ')
    return ''.join(buffer[:-1])

def toImg(x: torch.Tensor):
    x = (x / 2 + 0.5).clamp(0, 1)
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    x = (x * 255).round().astype("uint8")
    return x

def oneStep(
    i, t, latents, guidance_scale, 
    text_embeddings,  
):
    print('constrain...')
    latents = constrain(latents, i, True, True, .1)

    # print('expanding...')
    latents_expand = torch.cat([latents] * 2)
    sigma = scheduler.sigmas[i]
    latents_expand = latents_expand / (
        (sigma**2 + 1) ** 0.5
    )

    print('unet...')
    noise_pred: torch.Tensor = unet(
        latents_expand, t, 
        encoder_hidden_states=text_embeddings, 
    ).sample
    print('noise_pred:', noise_pred.shape)

    # print('linear guide...')
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # print('step...')
    return scheduler.step(noise_pred, i, latents).prev_sample

def constrain(
    z, i: int, mirror_not_avg, do_parallel, 
    guide_scale=1, 
):
    HALF = WIDTH // 2
    print('decoding...')
    dz: torch.Tensor = vae.decode(z).sample
    if do_parallel:
        print('encoding...')
        edz = vae.encode(dz).latent_dist.mean
    print('flipping...')
    sdz = dz.clone()
    if mirror_not_avg:
        if i % 2 == 0:
            sdz[:, :, :, :HALF] = sdz[:, :, :, HALF:].flip(dims=[3])
        else:
            sdz[:, :, :, HALF:] = sdz[:, :, :, :HALF].flip(dims=[3])
    else:
        sdz += sdz.flip(dims=[3])
        sdz *= .5
    print('encoding...')
    esdz = vae.encode(sdz).latent_dist.mean
    if do_parallel:
        return z + (esdz - edz) * guide_scale
    else:
        return esdz

with torch.no_grad():
    forward()
    visualize()
