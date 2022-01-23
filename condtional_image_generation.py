import torch
import torchvision

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
def main():
    model = torch.load("models/cGAN_best_model.pt")["model"].to(device)
    inputs = torch.zeros(64,dtype= torch.long).to(device)
    images = model(inputs,image_generation = True).detach().cpu()
    torchvision.utils.save_image(images.float(),
                                 'white.jpg' ,
                                nrow= 8,
                                 normalize=True)
    



if __name__ == '__main__':
    main()
    