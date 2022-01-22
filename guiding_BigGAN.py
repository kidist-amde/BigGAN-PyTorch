import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

def load_BigGAN_generator(config):
      # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
                
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  
  G = model.Generator(**config).cpu()
  utils.count_parameters(G)
  
  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)
  return G
def main():
    #parse command line and run    
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    G = load_BigGAN_generator(config)
    G_batch_size = max(config['G_batch_size'], config['batch_size']) 
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'])
    images = G(z_,G.shared(y_))  
    print(images.shape)
    images = torch.tensor(images.clone().detach())
    torchvision.utils.save_image(images.float(),
                                 'km.jpg' ,
                                nrow=int(G_batch_size**0.5),
                                 normalize=True)
    

if __name__ == '__main__':
    main()
    