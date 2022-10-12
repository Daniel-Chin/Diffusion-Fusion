'''
Subtracting horse seems useful!?!?
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

PROMPT = "a photo of traffic jam in a big city"
LADDER_LEN = 7
LADDER = torch.linspace(0, 1, LADDER_LEN)
WIDTH, HEIGHT = 512, 512

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
    num_inference_steps = 300
    guidance_scale = 7.5
    
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

    print('Sampling latent...')
    latents = torch.randn((
        1, unet.in_channels, HEIGHT // 8, WIDTH // 8,
    ), generator=generator)
    latents = latents.to(DEVICE)
    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.sigmas[0]
    latents = torch.cat([latents] * LADDER_LEN)

    with torch.autocast(DEVICE_STR) if (
        HAS_CUDA or torch.__version__ > '1.12.2'
    ) else nullcontext():
        for i, t in tqdm([*enumerate(scheduler.timesteps)], 'denoising'):
            print()
            latents = oneStep(
                i, t, latents, guidance_scale, 
                uncond_embeddings, text_embeddings,  
            )

    print('Saving...')
    with open('dwv.tensor', 'wb') as f:
        torch.save(latents, f)
    
def visualize():
    with open('dwv.tensor', 'rb') as f:
        latents = torch.load(f)
    
    latents = 1 / 0.18215 * latents

    print('Decoding image...')
    images = toImg(vae.decode(latents).sample)

    print('plotting...')
    DPI = 300
    matplotlib.rcParams['figure.dpi'] = DPI
    matplotlib.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(
        WIDTH * LADDER_LEN * 1.1 / DPI, 
        HEIGHT * 2 / DPI, 
    ))
    axes = fig.subplots(1, LADDER_LEN)
    fig.suptitle(PROMPT)
    for col_i in range(LADDER_LEN):
        ax = axes[col_i]
        img = images[col_i, :, :, :]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(format(LADDER[col_i].item(), '.2f'))
    plt.tight_layout()
    plt.subplots_adjust(wspace=.02)
    plt.savefig('out_dwv.png', dpi=DPI)
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
    uncond_embeddings, text_embeddings,  
):
    # print('expanding...')
    latents_expand = torch.cat([latents] * 2)
    sigma = scheduler.sigmas[i]
    latents_expand = latents_expand / (
        (sigma**2 + 1) ** 0.5
    )

    # predict the noise residual
    # print('widen text embedding...')
    text_embeddings = torch.cat(
        [uncond_embeddings] * LADDER_LEN + 
        [text_embeddings] * LADDER_LEN
    )
    print('unet...')
    noise_pred: torch.Tensor = unet(
        latents_expand, t, 
        encoder_hidden_states=text_embeddings, 
    ).sample
    print('noise_pred:', noise_pred.shape)

    # print('extract g...')
    uncond_np = noise_pred[0 * LADDER_LEN : 1 * LADDER_LEN]
    textcd_np = noise_pred[1 * LADDER_LEN : 2 * LADDER_LEN]
    guidance = textcd_np - uncond_np
    nudge = uncond_np + guidance * guidance_scale
    ms = nudge.square().mean()
    variation = torch.randn_like(nudge) * ms**.5
    print('ms', ms, variation.square().mean())
    variation *= LADDER.view(LADDER_LEN, 1, 1, 1)
    nudge += variation
    
    # print('step...')
    return scheduler.step(nudge, i, latents).prev_sample

with torch.no_grad():
    # forward()
    visualize()
