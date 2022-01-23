import functools
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
from vgg_face import Vgg_face_dag
from torch import nn
from torch.utils.data import Dataset,DataLoader

# Import my stuff
import inception_utils
import utils
import losses

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
class ConditionalBigGAN(nn.Module):
    def __init__(self,G,classifier,input_dim):
        super().__init__()
        self.G = G
        self.classifier = classifier 
        self.output_dim = G.dim_z
        self.input_generator = InputGenerator(input_dim,self.output_dim)
    def forward(self,inputs):
        onehot_inputs = nn.functional.one_hot(inputs).float().squeeze(1)
        mu_sigma = self.input_generator(onehot_inputs)
        mu = mu_sigma[:,:self.output_dim]
        sigma = mu_sigma[:,self.output_dim:]
        # to make sigma postive 
        sigma = nn.functional.softplus(sigma) 
        # generat random input z (epslon)
        eps = torch.randn(*mu.shape).to(device)
        z = mu+sigma * eps 
        # generat images 
        images = self.G(z,self.G.shared(inputs))  
        outputs = self.classifier(images)
        return outputs
class InputGenerator(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential( 
          nn.Linear(input_dim,256) ,
          nn.ReLU(),
          nn.Linear(256,512),
          nn.ReLU(),
          # input Genretor for mu and std (*2)
          nn.Linear(512,output_dim *2 )            
          )
    # forward pass
    def forward(self,inputs):
          return self.fc(inputs)
              
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
    batch_size = 64
    learning_rate = 1e-3
    epochs = 1000
    #parse command line and run    
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    G = load_BigGAN_generator(config)
    G_batch_size = max(config['G_batch_size'], config['batch_size']) 
    # z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
    #                          device=device, fp16=config['G_fp16'], 
    #                          z_var=config['z_var'])
    # images = G(z_,G.shared(y_))  
    # print(images.shape)
    # images = torch.tensor(images.clone().detach())
    # torchvision.utils.save_image(images.float(),
    #                              'km.jpg' ,
    #                             nrow=int(G_batch_size**0.5),
    #                              normalize=True)
    race_classifier = torch.load("models/race_classfier.pt")
    cGAN = ConditionalBigGAN(G,race_classifier,5)
    cGAN = cGAN.to(device)
    # freez model params
    for param in race_classifier.parameters():
          param.requires_grad=False
    for param in G.parameters():
        param.requires_grad=False         
    # generate input
    inputs = torch.randint(0,5,size=(32,))
    outputs = cGAN(inputs)
    print(outputs.shape)
    train_dataset = DummyDataset(20000)
    validation_dataset = DummyDataset(5000)
    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True,drop_last = True)
    val_loader = DataLoader(validation_dataset,batch_size = batch_size,shuffle = False,drop_last = False)
    optimizer =torch.optim.SGD(cGAN.input_generator.parameters(),lr = learning_rate,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    # traning loop
    for epoch in range(epochs):
      train_loss,train_acc = train_epoch(cGAN,train_loader,optimizer,criterion)
      val_loss,val_acc = evaluate(cGAN,val_loader,criterion)
      # log 
      print("Epoch:{}/{} Train_loss:{:.4f} Train_acc:{:.2f}%".format(epoch+1,epochs,train_loss,train_acc*100))
      print("val_loss:{:.4f} val_acc:{:.2f}%".format(val_loss,val_acc*100))
      
def get_race_classifier():
    model = Vgg_face_dag()
    state_dict = torch.load("vgg_face_dag.pth")
    model.load_state_dict(state_dict)
    # freez the other layer
    # for param in model.parameters():
    #       param.requires_grad = False
    model.fc6 = torch.nn.Linear(in_features=512, out_features=512, bias=True) 
    model.fc7 = torch.nn.Linear(in_features=512, out_features=1024, bias=True)
    model.fc8 = torch.nn.Linear(in_features=1024, out_features=5, bias=True)
    return model
class DummyDataset:
      def __init__(self,size):
          self.size =  size
      def __len__(self):
          return self.size
      def __getitem__(self,index):
          output = torch.randint(0,5,size=(1,))
          return output,output

def train_epoch(model,train_loader,optimizer,criterion):
    model.train()
    total = 0
    losses = 0
    corrects = 0
    for inputs,labels in tqdm(train_loader,total=len(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs,labels.squeeze(1))
        # batch total loss
        losses+=loss.item() * labels.size(0)
        total+=labels.size(0)
        predictions = outputs.argmax(dim = -1)
        # number of correctly predicted  class
        corrects+=(predictions==labels.squeeze(1)).float().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      # return avg loss and acc
    return losses/total , corrects/total
  
def evaluate(model,val_loader,criterion):
    model.eval()
    total = 0
    losses = 0
    corrects = 0
    # to skip backpropagation step
    with torch.no_grad():
      for inputs,labels in tqdm(val_loader,total=len(val_loader)):
          inputs = inputs.to(device)
          labels = labels.to(device)
          outputs = model(inputs)
          loss = criterion(outputs,labels.squeeze(1))
          losses+=loss.item() * labels.size(0)
          total+=labels.size(0)
          predictions = outputs.argmax(dim = -1)
          # number of correctly predicted  class
          corrects+=(predictions==labels.squeeze(1)).float().sum()
    # return avg loss and acc
    return losses/total , corrects/total

if __name__ == '__main__':
    main()
    