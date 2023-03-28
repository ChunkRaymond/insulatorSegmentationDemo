from utils.general import increment_path
from pathlib import Path
import os 
from tqdm import tqdm 
import torch
from torchvision import transforms
import cv2 as cv
import numpy as np 




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



def run(
        weights,
        ):

    model = torch.load(weights)
    model = model.to(device)
    model.eval()

    root_path = 'data/rgbImages'
    save_dir = increment_path('runs/detect/exp')
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm( os.listdir(root_path)):
        data = os.path.join(root_path,i)


        img_name = Path(data).name
        img = cv.imread(data)
        img = cv.resize(img, (160, 160))
        img = transform(img)
        img = torch.unsqueeze(img,0)
        img = img.to(device)
        

        output = torch.sigmoid(model(img))
        output = torch.squeeze(output).cpu().detach().numpy().copy().transpose(1,2,0)
        res_img = np.ones((160,160))*255

        for i in range(160):
            for j in range(160):
                if output[i,j,1] >=0.5:
                    res_img[i,j] = 0

        
        cv.imwrite(str(save_dir/img_name), res_img)
    


run('checkpoints/fcn_model_475.pt')