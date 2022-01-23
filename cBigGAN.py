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
        # resize images (32,32)generated_images to (224,224)race_classfier
        self.upsample = nn.UpsamplingNearest2d(size = 224)
    def forward(self,inputs,image_generation=False):
        onehot_inputs = nn.functional.one_hot(inputs,num_classes=5).float().squeeze(1)
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
        if image_generation:
            return images
        images = self.upsample(images)
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