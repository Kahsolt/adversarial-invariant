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
import pandas as pd


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
  tmodel = get_clf(args.tmodel)
  xmodel = get_vae(args.xmodel)
  print('[model]')
  print(f'   tmodel: {args.tmodel} ({param_cnt(tmodel)})')
  print(f'   xmodel: {args.xmodel} ({param_cnt(xmodel)})')
  tmodel = tmodel.eval().to(device)
  xmodel = xmodel.eval().to(device)

  tmodel.forward_original = tmodel.forward
  tmodel.forward = lambda X: tmodel.forward_original(normalize_imagenet_1k(X))
  xmodel.forward_original = xmodel.forward
  xmodel.forward = lambda X: xmodel.forward_original(X).sample.clamp(0.0, 1.0)

  dataset = ImageNet_1k(DATA_PATH)
  data_gen = gen_batch_data(tmodel, dataset, args.steps, args.eps, args.alpha)

  model_version = f'version_{args.version}'
  MODEL_PATH = LOG_PATH / model_version
  ckpt_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.ckpt')]
    
  df = pd.DataFrame(columns = ['checkpoint', 'L1_X_VAEAX', 'L1_X_VAEX', 'L1_VAEX_VAEAX']);
  
  for ckpt_file in ckpt_files:
    ckpt = torch.load(MODEL_PATH / ckpt_file, map_location=device)
    xmodel.load_state_dict(ckpt['model'])
    del ckpt
    xmodel.eval()
    print('Loading model from', MODEL_PATH / ckpt_file)
        
    L1_X_VAEAX, L1_X_VAEX, L1_VAEX_VAEAX = None, None, None
    for AX, X, Y in data_gen():
      VAE_X = xmodel(X)
      VAE_AX = xmodel(AX)
      #breakpoint()
      l1_x_vaeax = F.l1_loss(X.flatten(1), VAE_AX.flatten(1), reduction='none').mean(-1).cpu().detach().numpy()
      l1_x_vaex = F.l1_loss(X.flatten(1), VAE_X.flatten(1), reduction='none').mean(-1).cpu().detach().numpy()
      l1_vaex_vaeax = F.l1_loss(VAE_X.flatten(1), VAE_AX.flatten(1), reduction='none').mean(-1).cpu().detach().numpy()
            
      L1_X_VAEAX = stack_numpy(L1_X_VAEAX, l1_x_vaeax)
      L1_X_VAEX = stack_numpy(L1_X_VAEX, l1_x_vaex)
      L1_VAEX_VAEAX = stack_numpy(L1_VAEX_VAEAX, l1_vaex_vaeax)
      
    L1_X_VAEAX, L1_X_VAEX, L1_VAEX_VAEAX = np.mean(L1_X_VAEAX), np.mean(L1_X_VAEX), np.mean(L1_VAEX_VAEAX)
    
    df.loc[len(df)] = [ckpt_file, L1_X_VAEAX, L1_X_VAEX, L1_VAEX_VAEAX]
    print(f'X and VAE(AX): {L1_X_VAEAX:.4f}, X and VAE(X): {L1_X_VAEX:.4f}, VAE(X) and VAE(AX): {L1_VAEX_VAEAX:.4f}')

  print(df)
  df.to_csv('data.csv', index=False)
    

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', '--tmodel',     type=str,  default='resnet50',                  choices=PRETRAINED_MODELS, help='victim model under PGD attack')
  parser.add_argument('-X', '--xmodel',     type=str,  default='stabilityai/sd-vae-ft-mse', choices=PRETRAINED_HUBS,   help='feature extractor for adversarial-invariant outputs')
  parser.add_argument('-V', '--version',    type=int,  default=0)
  parser.add_argument('-B', '--batch_size', type=int,  default=4,     help='extractor training batch_size')
  parser.add_argument('--steps',            type=int,  default=20,    help='PGD attack steps')
  parser.add_argument('--eps',              type=eval, default=8/255, help='PGD attack eps')
  parser.add_argument('--alpha',            type=eval, default=1/255, help='PGD attack alpha')
  args = parser.parse_args()
    
  run(args)
  