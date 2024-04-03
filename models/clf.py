#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03

from utils import *

import torchvision.models as M

# pretrained input size: 224 x 224
PRETRAINED_MODELS = {
  'resnet18',
  'resnet34',
  'resnet50',
  'resnet101',
  'resnet152',
  'resnext50_32x4d',
  'resnext101_32x8d',
  'resnext101_64x4d',
  'wide_resnet50_2',
  'wide_resnet101_2',

  'densenet121',
  'densenet161',
  'densenet169',
  'densenet201',

  'squeezenet1_0',
  'squeezenet1_1',
  'mobilenet_v2',
  'mobilenet_v3_large',
  'mobilenet_v3_small',
  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',

  'vit_b_16',
  'vit_b_32',
  'vit_l_16',
  'vit_l_32',
  'vit_h_14',
  'swin_t',
  'swin_s',
  'swin_b',
  'maxvit_t',
}


def list_clf():
  print(f'[classifers]')
  for name in PRETRAINED_MODELS:
    print(f'  {name}')


def get_clf(name:str) -> Model:
  model_cls = getattr(M, name)
  return model_cls(pretrained=True)


if __name__ == '__main__':
  print('[classifers]')
  for name in [
    'resnet50',
    'vit_b_16',
  ]:
    model = get_clf(name)
    print(f'  {name}: {param_cnt(model)}')
