#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03 

from torch.autograd import grad

from utils import *


last_pgd_setting: Tuple[int, float, float] = None

def resolve_real_settings(default_steps:int, default_eps:float, default_alpha:float):
  # Allow reading settings from envvar to enable dynamic control :)
  global last_pgd_setting
  steps = get_envvar('PGD_STEPS', type=int,   default=default_steps)
  eps   = get_envvar('PGD_EPS',   type=float, default=default_eps)
  alpha = get_envvar('PGD_ALPHA', type=float, default=default_alpha)
  new_pgd_setting = (steps, eps, alpha)
  if last_pgd_setting != new_pgd_setting:
    print(f'>> PGD setting changed: {last_pgd_setting} => {new_pgd_setting}')
    last_pgd_setting = new_pgd_setting
  return new_pgd_setting


def pgd(model:Model, X:Tensor, Y:Tensor=None, steps:int=20, eps:float=8/255, alpha:float=1/255) -> Tensor:
  # resolve the real setting
  steps, eps, alpha = resolve_real_settings(steps, eps, alpha)

  # decide Y if not provided
  if Y is None:
    with torch.inference_mode():
      Y = model(X).argmax(dim=-1)

  # prevent data pollution
  X = X.detach().clone()
  Y = Y.detach().clone()

  # init noise
  with torch.no_grad():
    AX = X.detach().clone()
    AX = AX + torch.empty_like(AX).uniform_(-eps, eps)
    AX = AX.clamp(0.0, 1.0).detach()

  # pgd iter
  for _ in range(steps):
    with torch.enable_grad():
      AX.requires_grad = True
      logits = model(AX)
      loss = F.cross_entropy(logits, Y, reduction='none')
      g = grad(loss, AX, loss)[0]

    with torch.no_grad():
      AX = AX.detach() + g.sign() * alpha
      AX = X + (AX - X).clamp(-eps, eps)
      AX = AX.clamp(0.0, 1.0).detach()

  return AX


if __name__ == '__main__':
  from models.clf import get_clf
  from data import normalize_imagenet_1k

  model = get_clf('resnet18')
  X = torch.rand([16, 3, 224, 224])
  print('X:', X.shape)

  AX = pgd(model, X)
  print('AX:', AX.shape)
  L1 = torch.abs(AX - X)
  print('Linf:', L1.max().item())
  print('L1:', L1.mean().item())

  with torch.inference_mode():
    pred_X  = model(normalize_imagenet_1k( X)).argmax(dim=-1)
    pred_AX = model(normalize_imagenet_1k(AX)).argmax(dim=-1)
    tot = len(pred_X)
    ok = (pred_X == pred_AX).sum().item()
    print(f'Acc: {ok} / {tot} = {ok / tot:.5%}')
