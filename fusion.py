'''
Subtracting horse seems useful!?!?
pixel-level constraints
    symmetry
    横看成岭侧成峰
'''

from contextlib import nullcontext

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from winsound import Beep
logging.set_verbosity_error()

from shared import *

LADDER_LEN = 9
LADDER = torch.linspace(0, 1, LADDER_LEN)

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

def main():
    prompt_pair = [
        "a photo of a person doing a handstand on a horse", 
        "a photo of an astronaut riding a horse on mars", 
        # "a photo of a horse", 
    ]
    num_inference_steps = 20
    guidance_scale = 7.5
    width, height = 512, 512
    
    print('Encoding text...')
    text_input_pair = [tokenizer(
        x, padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt", 
    ) for x in prompt_pair]
    text_embeddings_pair = [text_encoder(
        x.input_ids.to(DEVICE), 
    )[0] for x in text_input_pair]

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
        1, unet.in_channels, height // 8, width // 8,
    ), generator=generator)
    latents = latents.to(DEVICE)
    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.sigmas[0]
    latents = torch.cat([latents] * LADDER_LEN)

    with torch.autocast(DEVICE_STR) if (
        HAS_CUDA or torch.__version__ > '1.10.0'
    ) else nullcontext():
        for i, t in tqdm([*enumerate(scheduler.timesteps)], 'denoising'):
            latents = oneStep(
                i, t, latents, guidance_scale, 
                uncond_embeddings, text_embeddings_pair,  
            )

    print('Decoding image...')
    latents = 1 / 0.18215 * latents
    fusion_images = toImg(vae.decode(latents).sample)
    interp_images = torch.zeros_like(fusion_images)
    for i, k in enumerate(LADDER):
        interp_images[i, :, :, :] = toImg(vae.decode(
            latents[0,  :, :, :] * (1 - k) +
            latents[-1, :, :, :] * k
        ).sample)

    fig, axes = plt.subplots(2, LADDER_LEN)
    for row_i, method_name in enumerate(('fusion', 'interp')):
        axes[row_i][LADDER_LEN // 2].set_title(method_name)
        for col_i in range(LADDER_LEN):
            ax = axes[row_i][col_i]
            img = locals()[method_name + '_images'][col_i, :, :, :]
            ax.imshow(img)
    Beep(330, 2)
    plt.show()
    from console import console
    console({**globals(), **locals()})

def toImg(x: torch.Tensor):
    x = (x / 2 + 0.5).clamp(0, 1)
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    x = (x * 255).round().astype("uint8")

def oneStep(
    i, t, latents, guidance_scale, 
    uncond_embeddings, text_embeddings_pair,  
):
    latents_expand = torch.cat([latents] * 3)
    sigma = scheduler.sigmas[i]
    latents_expand = latents_expand / (
        (sigma**2 + 1) ** 0.5
    )

    # predict the noise residual
    text_embeddings = torch.cat(
        [uncond_embeddings] * LADDER_LEN + 
        [text_embeddings_pair[0]] * LADDER_LEN + 
        [text_embeddings_pair[1]] * LADDER_LEN
    )
    noise_pred: torch.Tensor = unet(
        latents_expand, t, 
        encoder_hidden_states=text_embeddings, 
    ).sample

    # perform guidance
    g0 = (
        noise_pred[1 * LADDER_LEN : 2 * LADDER_LEN, :, :, :] - 
        noise_pred[0 * LADDER_LEN : 1 * LADDER_LEN, :, :, :]
    )
    g1 = (
        noise_pred[2 * LADDER_LEN : 3 * LADDER_LEN, :, :, :] - 
        noise_pred[0 * LADDER_LEN : 1 * LADDER_LEN, :, :, :]
    )
    # gn = nn - noise_pred_uncond
    # noise_pred = noise_pred_uncond + (g0 + g1 - gn) * guidance_scale
    g = g0 + g1
    my_scale = max(g0.norm(), g1.norm()) / g.norm()
    noise_pred = (
        noise_pred[0 * LADDER_LEN : 1 * LADDER_LEN] 
        + g * guidance_scale * my_scale
    )
    # print('\n', 
    #     g0.norm().item(), 
    #     g1.norm().item(), 
    #     (g0 + g1).norm().item() / 2**.5, 
    # )

    # compute the previous noisy sample x_t -> x_t-1
    return scheduler.step(noise_pred, i, latents).prev_sample

with torch.no_grad():
    main()
