from argparse import ArgumentParser
from traceback import print_exc

from PIL import Image
import tqdm

from data import DataLoader, Dataset, ImageNet_1k, normalize_imagenet_1k
from models import get_clf, get_vae, PRETRAINED_HUBS
from attack import pgd
from utils import *
from plot import plot2

IMAGE_PATH = BASE_PATH / 'image_vae'


def tensor_to_image(tensor):
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.detach().cpu()
    image = tensor.numpy()
    image = Image.fromarray((image * 255).astype('uint8'))
    return image


def save_tensor_image(tensor, file_path):
    image = tensor_to_image(tensor)
    image.save(file_path)
        

def initialize_image_saving(directory, base_filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    counter = [0] 

    def save_batch_images(batch_tensors, format='png'):
        nonlocal counter
        for tensor in batch_tensors:
            file_path = os.path.join(directory, f"{base_filename}_{counter[0]}.{format}")
            save_tensor_image(tensor, file_path)
            print(f"Image saved to {file_path}")
            counter[0] += 1

    return save_batch_images

        
        
def main(args):
    xmodel = get_vae(args.xmodel)
    xmodel = xmodel.eval().to(device)
    xmodel.forward_original = xmodel.forward
    xmodel.forward = lambda X: xmodel.forward_original(X).sample
  
    dataset = ImageNet_1k(DATA_PATH)
    if args.single == -1:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=1, drop_last=True, pin_memory=True)
    save_images = initialize_image_saving(directory=IMAGE_PATH, base_filename="image")
    
    if args.single == -1:
        for X, Y in dataloader:
            X: Tensor = X.to(device)
            Y: Tensor = Y.to(device)

            pred = xmodel(X)
            save_images(pred)
            
        print(f"Images saved to {IMAGE_PATH}")
    else:
        for i, (X, Y) in enumerate(dataloader):
            if i == args.single:
                X: Tensor = X.to(device)
                Y: Tensor = Y.to(device)

                pred = xmodel(X)
                plot2(X, pred)
                break
        
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-X', '--xmodel',     type=str,  default='madebyollin/taesd', choices=PRETRAINED_HUBS,   help='feature extractor for adversarial-invariant outputs')
    parser.add_argument('-B', '--batch_size', type=int,  default=4,     help='extractor training batch_size')
    parser.add_argument('-S', '--single',     type=int,  default=-1,    help='single image index')
    args = parser.parse_args()
    
    main(args)