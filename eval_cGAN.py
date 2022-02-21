from tqdm.auto import tqdm
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
from cBigGAN import ConditionalBigGAN,InputGenerator
from sklearn.metrics import classification_report

# Import my stuff
import inception_utils
import utils
import losses
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
def main():
  # load the model
  cGAN_model = torch.load("models/cGAN_best_model.pt",map_location = device)["model"]
  actual = []
  predicted = []
  for i in tqdm(range(1000),total = 1000):
  # generate random i/p
    inputs = torch.randint(0,5,size =(64,1)).to(device)
    outputs = cGAN_model(inputs).argmax(dim = -1)
    actual.extend(inputs.view(-1).tolist())
    predicted.extend(outputs.view(-1).tolist())
 
  print(classification_report(actual, predicted, target_names =[ "White", "Black", "Asian", "Indian",  "Others"],labels=[0,1,2,3,4]))
  











if __name__ == '__main__':
    main()
    