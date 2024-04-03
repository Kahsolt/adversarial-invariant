#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03

import json
from pathlib import Path
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

IMAGENET_STAT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STAT_STD  = (0.229, 0.224, 0.225)


class ImageNet_1k(Dataset):

  def __init__(self, root:Path):
    base_dp = root / 'imagenet-1k'
    fps = list((base_dp / 'val').iterdir())
    with open(base_dp /'image_name_to_class_id_and_name.json', encoding='utf-8') as fh:
      mapping = json.load(fh)
    tgts = [mapping[fp.name]['class_id'] for fp in fps]
    self.metadata = list(zip(fps, tgts))

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp).convert('RGB')
    im = TF.to_tensor(img)
    return im, tgt


def normalize_imagenet_1k(X:Tensor) -> Tensor:
  return TF.normalize(X, IMAGENET_STAT_MEAN, IMAGENET_STAT_STD)
