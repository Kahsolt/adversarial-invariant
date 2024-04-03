#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03

from __future__ import annotations
import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

import os
from pathlib import Path
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module as Model

torch.set_float32_matmul_precision('high')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  torch.backends.cuda.matmul.allow_tf32 = True

BASE_PATH = Path(__file__).parent.absolute()
DATA_PATH = BASE_PATH / 'data'
LOG_PATH = BASE_PATH / 'log'


def param_cnt(model:Model, only_trainable:bool=False) -> int:
  return sum([p.numel() for p in model.parameters() if (not only_trainable or p.requires_grad)])


def get_envvar(name:str, type:type=str, default:Any=None) -> Any:
  var = os.getenv(name)
  if var is None: return default
  return type(var)


class ValueWindow:

  def __init__(self, nlen:int=20):
    self.nlen = nlen
    self.values = []

  @classmethod
  def from_list(cls, values:List[Any], nlen:int=20) -> ValueWindow:
    wv = ValueWindow(nlen)
    wv.values = values
    return wv

  def append(self, v:Any):
    self.values.append(v)
  
  def __len__(self):
    return len(self.values)

  @property
  def sum(self):
    return sum(self.values)

  @property
  def mean(self):
    return (sum(self.values) / len(self.values)) if len(self.values) else 0.0

  @property
  def recent_sum(self, nlen:int=None):
    nlen = nlen or self.nlen
    recent_values = self.values[-nlen:]
    return sum(recent_values)

  @property
  def recent_mean(self, nlen:int=None):
    nlen = nlen or self.nlen
    recent_values = self.values[-nlen:]
    return (sum(recent_values) / len(recent_values)) if len(recent_values) else 0.0

  @property
  def last_value(self):
    return self.values[-1]
