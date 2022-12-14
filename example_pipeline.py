from contextlib import nullcontext

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from tqdm.auto import tqdm
from PIL import Image
logging.set_verbosity_error()

from shared import *

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

tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14", 
)
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, 
    beta_schedule="scaled_linear", num_train_timesteps=1000, 
)

def main():
    prompt = "a photo of a person doing a handstand on a horse"
    num_inference_steps = 20
    guidance_scale = 7.5
    width, height = 512, 512
    batch_size = 1
    
    text_input = tokenizer(
        prompt, padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt", 
    )
    text_embeddings = text_encoder(
        text_input.input_ids.to(DEVICE), 
    )[0]

    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", 
        max_length=tokenizer.model_max_length, 
        return_tensors="pt", 
    )
    uncond_embeddings = text_encoder(
        uncond_input.input_ids.to(DEVICE)
    )[0]

    text_embeddings = torch.cat([
        uncond_embeddings, text_embeddings, 
    ])

    latents = torch.randn((
        batch_size, unet.in_channels, height // 8, width // 8,
    ))
    latents = latents.to(DEVICE)
    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.sigmas[0]

    with torch.autocast(DEVICE_STR) if (
        HAS_CUDA or torch.__version__ > '1.10.0'
    ) else nullcontext():
        for i, t in tqdm([*enumerate(scheduler.timesteps)]):
            latents = oneStep(
                i, t, latents, guidance_scale, text_embeddings, 
            )

    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    # pil_images[0].save(prompt[0].replace(' ', '_') + '.png')
    pil_images[0].save('1.png')

def oneStep(
    i, t, latents, guidance_scale, text_embeddings, 
):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    sigma = scheduler.sigmas[i]
    latent_model_input = latent_model_input / (
        (sigma**2 + 1) ** 0.5
    )

    # predict the noise residual
    with torch.no_grad():
        noise_pred: torch.Tensor = unet(
            latent_model_input, t, 
            encoder_hidden_states=text_embeddings, 
        ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        return scheduler.step(noise_pred, i, latents).prev_sample

with torch.no_grad():
    main()
