#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03

import sys
import json
from argparse import ArgumentParser
from traceback import print_exc

from torch.optim import Adam
from tensorboardX import SummaryWriter

from data import DataLoader, Dataset, ImageNet_1k, normalize_imagenet_1k
from models import get_clf, get_vae, PRETRAINED_MODELS, PRETRAINED_HUBS
from attack import pgd
from utils import *


def get_exp_name() -> str:
  idx = 0
  while True:
    exp_name = f'version_{idx}'
    if not (LOG_PATH / exp_name).exists():
      return exp_name
    idx += 1


def gen_batch_data(model:Model, dataset:Dataset, steps:int=20, eps:float=8/255, alpha:float=1/255):
  def wrapper():
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    for X, Y in dataloader:
      X: Tensor = X.to(device, non_blocking=True)
      Y: Tensor = Y.to(device, non_blocking=True)
      AX = pgd(model, X, Y, steps, eps, alpha)
      yield AX, X, Y
  return wrapper


def run(args):
  ''' Log '''
  exp_name = args.exp_name or get_exp_name()
  log_dp = LOG_PATH / exp_name
  sw = SummaryWriter(log_dp)
  with open(log_dp / 'hparam.json', 'w', encoding='utf-8') as fh:
    hp = {
      'cmd': ' '.join(sys.argv),
      'args': vars(args),
    }
    json.dump(hp, fh, indent=2, ensure_ascii=False)

  ''' Model & Optim '''
  tmodel = get_clf(args.tmodel)
  xmodel = get_vae(args.xmodel)
  print('[model]')
  print(f'   tmodel: {args.tmodel} ({param_cnt(tmodel)})')
  print(f'   xmodel: {args.xmodel} ({param_cnt(xmodel)})')
  tmodel = tmodel.eval() .to(device)
  xmodel = xmodel.train().to(device)

  optim = Adam(xmodel.parameters(), lr=args.lr)

  ''' Monkey-Patching model forward_fn '''
  tmodel.forward_original = tmodel.forward
  tmodel.forward = lambda X: tmodel.forward_original(normalize_imagenet_1k(X))
  xmodel.forward_original = xmodel.forward
  xmodel.forward = lambda X: xmodel.forward_original(X).sample.clamp(0.0, 1.0)

  ''' Ckpt & Bookeep '''
  if args.load:
    ckpt = torch.load(args.load, map_location=device)
    xmodel.load_state_dict(ckpt['model'])
    optim .load_state_dict(ckpt['optim'])
    epochs  = ckpt['epochs']
    steps   = ckpt['steps']
    loss_wv = ValueWindow.from_list(ckpt['loss_wv'], nlen=20)
    asr_wv  = ValueWindow.from_list(ckpt['asr_wv'],  nlen=20)  
  else:
    epochs  = 1
    steps   = 0
    loss_wv = ValueWindow(nlen=20)
    asr_wv  = ValueWindow(nlen=20)

  ''' Data '''
  dataset = ImageNet_1k(DATA_PATH)
  data_gen = gen_batch_data(tmodel, dataset, args.steps, args.eps, args.alpha)

  ''' Train '''
  try:
    for epochs in range(epochs, args.epochs+1):
      for AX, X, Y in data_gen():
        # tmodel for ASR
        with torch.inference_mode():
          pred = tmodel(AX).argmax(dim=-1)
          racc = (Y == pred).sum().item() / len(pred)
          asr_wv.append(1 - racc)

        # xmodel for loss
        optim.zero_grad()
        fmap_AX = xmodel(AX)
        fmap_X  = xmodel(X)
        loss = F.huber_loss(fmap_AX, fmap_X)
        loss.backward()
        optim.step()

        loss_wv.append(loss.item())

        steps += 1

        if steps % args.log_interval == 0:
          print(f'[step {steps}] asr: {asr_wv.recent_mean}, loss: {loss_wv.recent_mean}')

        if steps % args.summary_interval == 0:
          sw.add_scalar('loss', loss_wv.last_value, global_step=steps)
          sw.add_scalar('asr',  asr_wv.last_value,  global_step=steps)
          sw.flush()

        if steps % args.plot_interval == 0:
          sw.add_image('fmap_AX', fmap_AX[0], global_step=steps, dataformats='CHW')
          sw.add_image('fmap_X',  fmap_X[0],  global_step=steps, dataformats='CHW')
          sw.flush()

        if steps % args.ckpt_interval == 0:
          ckpt = {
            'model': xmodel.state_dict(),
            'optim': optim.state_dict(),
            'epochs': epochs,
            'steps': steps,
            'loss_wv': loss_wv.values,
            'asr_wv': asr_wv.values,
          }
          torch.save(ckpt, log_dp / f'steps-{steps}.ckpt')

  except KeyboardInterrupt:
    print('>> Exit by Ctrl+C')
  except:
    print_exc()

  ''' Save '''
  ckpt = {
    'model': xmodel.state_dict(),
    'optim': optim.state_dict(),
    'epochs': epochs,
    'steps': steps,
    'loss_wv': loss_wv.values,
    'asr_wv': asr_wv.values,
  }
  torch.save(ckpt, log_dp / f'steps-{steps}.ckpt')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', '--tmodel',     type=str,  default='resnet50',                  choices=PRETRAINED_MODELS, help='victim model under PGD attack')
  parser.add_argument('-X', '--xmodel',     type=str,  default='stabilityai/sd-vae-ft-mse', choices=PRETRAINED_HUBS,   help='feature extractor for adversarial-invariant outputs')
  parser.add_argument('-E', '--epochs',     type=int,  default=500,   help='extractor training epochs')
  parser.add_argument('-B', '--batch_size', type=int,  default=4,     help='extractor training batch_size')
  parser.add_argument('-lr', '--lr',        type=eval, default=2e-6,  help='extractor training lr')
  parser.add_argument('-L', '--load',       type=Path,                help='resume from ckpt')
  parser.add_argument('--steps',            type=int,  default=20,    help='PGD attack steps')
  parser.add_argument('--eps',              type=eval, default=8/255, help='PGD attack eps')
  parser.add_argument('--alpha',            type=eval, default=1/255, help='PGD attack alpha')
  parser.add_argument('--log_interval',     type=int,  default=5)
  parser.add_argument('--summary_interval', type=int,  default=10)
  parser.add_argument('--plot_interval',    type=int,  default=100)
  parser.add_argument('--ckpt_interval',    type=int,  default=2000)
  parser.add_argument('--exp_name', help='assign experiment name')
  args, _ = parser.parse_known_args()

  run(args)
