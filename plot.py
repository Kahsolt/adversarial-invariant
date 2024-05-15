from utils import *
import matplotlib.pyplot as plt

def plot2(X:Tensor, output_X:Tensor):
  # X.AX.DX.shape: [B=1, C=3, H=224, W224]
  X = X.to('cpu')
  output_X = output_X.to('cpu')
  
  x  = X.squeeze(dim=0).permute([1, 2, 0]).detach().numpy()
  output = output_X.squeeze(dim=0).permute([1, 2, 0]).detach().numpy()
  
  fig, axes = plt.subplots(1, 2)
  axes[0].imshow(x)
  axes[0].set_title('input picture')
  axes[1].imshow(output)
  axes[1].set_title('output picture')

  plt.show()