from tqdm.auto import tqdm
import torch
import numpy as np
import glob
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from vgg_face import Vgg_face_dag, vgg_face_dag
import torchvision.transforms as transforms
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

class UTKfaceDataset(Dataset):
  def __init__(self, image_paths,transform=None):
    self.transform = transform
    self.image_paths = image_paths
 
  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    image_path = self.image_paths[index]
    # race lable is 3 whixh is at index 2
    #The labels of each face image is embedded in the file name, formated like
    # [age]_[gender]_[race]_[date&time].jpg
    try:

      target = int(image_path.split("_")[2])
    except:
      target = 4
    # read image and convert to RGB
    img = Image.open(image_path).convert("RGB")

    if self.transform is not None:
      img = self.transform(img)

    return img, target
      
  def __len__(self):
      return len(self.image_paths)
def main():
    norm_mean = [0.5,0.5,0.5]
    norm_std = [0.5,0.5,0.5]
    batch_size = 64
    learning_rate = 1e-3
    epochs = 20
    images_folder = "data/UTKFace/"
    image_paths = []
    for file_path in glob.glob(images_folder + "*.jpg"):
        image_paths.append(file_path)
    train_paths,val_paths = train_test_split(image_paths,test_size = 0.2)
    # data augmentation
    train_transform = [transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip()]
    train_transform = transforms.Compose(train_transform + [
                     transforms.ToTensor(),
                     transforms.Normalize(norm_mean, norm_std)])
    
    val_transform = transforms.Compose([transforms.Resize(32),
                     transforms.ToTensor(),
                     transforms.Normalize(norm_mean, norm_std)])
    
    train_dataset = UTKfaceDataset(train_paths,train_transform)
    validation_dataset = UTKfaceDataset(val_paths,val_transform)
    print(len(train_dataset))
    print(len(validation_dataset))
    
    model = Vgg_face_dag()
    state_dict = torch.load("vgg_face_dag.pth")
    model.load_state_dict(state_dict)
    # freez the other layer
    for param in model.parameters():
          param.requires_grad = False
    model.fc6 = torch.nn.Linear(in_features=512, out_features=512, bias=True) 
    model.fc7 = torch.nn.Linear(in_features=512, out_features=1024, bias=True)
    model.fc8 = torch.nn.Linear(in_features=1024, out_features=5, bias=True)
    model = model.to(device)
    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True,drop_last = True)
    val_loader = DataLoader(validation_dataset,batch_size = batch_size,shuffle = False,drop_last = False)
    params = list(model.fc8.parameters()) +list(model.fc7.parameters()) +list(model.fc6.parameters())
    optimizer =torch.optim.SGD(params,lr = learning_rate,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    # save the best model
    best_acc = 0
    # traning loop
    for epoch in range(epochs):
      train_loss,train_acc = train_epoch(model,train_loader,optimizer,criterion)
      val_loss,val_acc = evaluate(model,val_loader,criterion)
      # log 
      print("Epoch:{}/{} Train_loss:{:.4f} Train_acc:{:.2f}%".format(epoch+1,epochs,train_loss,train_acc*100))
      print("val_loss:{:.4f} val_acc:{:.2f}%".format(val_loss,val_acc*100))
      # save the best model
      if val_acc>best_acc:
            best_acc = val_acc
            torch.save({"best_acc":best_acc,"model":model,"epoch":epoch},"best_model.pt")
def train_epoch(model,train_loader,optimizer,criterion):
    model.train()
    total = 0
    losses = 0
    corrects = 0
    for inputs,labels in tqdm(train_loader,total=len(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        # batch total loss
        losses+=loss.item() * labels.size(0)
        total+=labels.size(0)
        predictions = outputs.argmax(dim = -1)
        # number of correctly predicted  class
        corrects+=(predictions==labels).float().sum()
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
          loss = criterion(outputs,labels)
          losses+=loss.item() * labels.size(0)
          total+=labels.size(0)
          predictions = outputs.argmax(dim = -1)
          # number of correctly predicted  class
          corrects+=(predictions==labels).float().sum()
    # return avg loss and acc
    return losses/total , corrects/total
    
      
if __name__ == '__main__':
    main()
    