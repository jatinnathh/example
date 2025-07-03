import torch
import numpy as np
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str = None, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5,
             sampler_name="ddpm", n_inference_steps=50, models={}, seed=None, device=None, idle_device=None, tokenizer=None,callback=None, cancel_flag=None):

    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        clip = models["clip"].to(device)

        if do_cfg:
            cond_tokens = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
            uncond_tokens = tokenizer(uncond_prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)

            cond_context = clip(cond_tokens)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context], dim=0)
        else:
            tokens = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
            context = clip(tokens)

        if idle_device:
            clip.to(idle_device)

        if sampler_name == "ddpm":
            from .ddpm import DDPMSampler
            sampler = DDPMSampler(generator)
        
        else: 
            raise ValueError(f"Unknown sampler: {sampler_name}")

        sampler.set_inference_timesteps(n_inference_steps)
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"].to(device)
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = torch.tensor(np.array(input_image_tensor), dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1)).unsqueeze(0).permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            encoder.to(idle_device)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"].to(device)
        timesteps = tqdm(sampler.timesteps)

        for i,timestep in enumerate(timesteps):
            if cancel_flag and cancel_flag():
                return None
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                if model_output.shape[0] != 2:
                    raise RuntimeError(f"Expected model output shape [2, ...], got {model_output.shape}")
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)
            if callback:
                callback(i + 1, len(sampler.timesteps))

        diffusion.to(idle_device)

        decoder = models["decoder"].to(device)
        images = decoder(latents)
        decoder.to(idle_device)

        images = rescale(images, (-1, 1), (0, 255), clamp=True).permute(0, 2, 3, 1)
        return images.to("cpu", torch.uint8).numpy()[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)