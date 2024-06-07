import torch
import os
import torchvision.transforms as T
from PIL import Image
import pickle

dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dinov2_vitl14.to(device)
dinov2_vitl14.eval() 

patch_size = dinov2_vitl14.patch_size #14

patch_h = 980//patch_size
patch_w = 980//patch_size

feat_dim = 128 

#dinov2 모델 이용해서 feature map 뽑기
folder_path = "./keyframes/"
total_features = []

trans = T.Compose([
    T.CenterCrop(980),
    T.ToTensor(),
    T.Normalize(mean=0.5, std=0.2)
])

with torch.no_grad():
    for img_path in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_path)
        img = Image.open(img_path).convert('RGB')
        img_t = trans(img) #(1, 980, 980)

        feature_dict = dinov2_vitl14.forward_features(img_t.unsqueeze(0).to(device))
        features = feature_dict['x_norm_patchtokens'] #(1, 4900, 1024) -> 4900(70*70) 이유: img size / patch size (980/14)
        total_features.append(features)

total_features = torch.cat(total_features, dim=0).cpu() #(image 개수, 4900, 1024)

with open('dinov2_embedding.pickle', 'wb') as f:
        pickle.dump(total_features, f, protocol=4)