from .clip import CLIP
from .encoder import VAE_Encoder
from .decoder import VAE_Decoder
from .diffusion import Diffusion

from . import model_converter


def preload_models_from_standard_weights(ckpt_path, device):
    # Load the converted state dict
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # Safely extract submodule weights
    encoder_weights = state_dict.get("encoder", {})
    decoder_weights = state_dict.get("decoder", {})
    diffusion_weights = state_dict.get("diffusion", {})
    clip_weights = state_dict.get("clip", {})

    # Initialize models
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    diffusion = Diffusion().to(device)
    clip = CLIP().to(device)

    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_weights, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    decoder.load_state_dict(decoder_weights, strict=False)
    diffusion.load_state_dict(diffusion_weights, strict=False)
    clip.load_state_dict(clip_weights, strict=False)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
