#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03

from utils import *

from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders import AutoencoderKL, AutoencoderTiny, AsymmetricAutoencoderKL, ConsistencyDecoderVAE
from diffusers.models.vq_model import VQModel
from diffusers.pipelines import VQDiffusionPipeline

# pretrained input size: 512 x 512 | 768 x 768
PRETRAINED_ARCH_TO_HUBS = {
  'AutoencoderKL': [
    'stabilityai/sd-vae-ft-ema',
    'stabilityai/sd-vae-ft-mse',
    'stabilityai/sdxl-vae',
    'madebyollin/sdxl-vae-fp16-fix',
  ],
  'AutoencoderTiny': [
    'madebyollin/taesd',
    'madebyollin/taesdxl',
  ],
  'AsymmetricAutoencoderKL': [
    'cross-attention/asymmetric-autoencoder-kl-x-1-5',
    'cross-attention/asymmetric-autoencoder-kl-x-2',
  ],
  'ConsistencyDecoderVAE': [
    'openai/consistency-decoder',
  ],
  'VQModel': [
    'microsoft/vq-diffusion-ithq',    # this is deprecated :(
  ],
}
PRETRAINED_HUBS = [hub for hub_list in PRETRAINED_ARCH_TO_HUBS.values() for hub in hub_list]


def list_vae():
  for arch, hub_list in PRETRAINED_ARCH_TO_HUBS.items():
    print(f'[{arch}]')
    for hub in hub_list:
      print(f'  {hub}')


def _get_arch(hub:str):
  for arch, hub_list in PRETRAINED_ARCH_TO_HUBS.items():
    if hub in hub_list:
      return arch
  raise ValueError(f'>> unknown hub: {hub}')


def get_vae(repo:str) -> Model:
  arch = _get_arch(repo)
  if arch == 'VQModel':
    return VQDiffusionPipeline.from_pretrained(repo)
  arch_cls: ModelMixin = globals()[arch]
  return arch_cls.from_pretrained(repo)


if __name__ == '__main__':
  print('[autoencoders]')
  for hub in [
    'stabilityai/sd-vae-ft-mse',
    'madebyollin/sdxl-vae-fp16-fix',
    'madebyollin/taesd',
    'madebyollin/taesdxl',
  ]:
    model = get_vae(hub)
    print(f'  {hub}: {param_cnt(model)}')
