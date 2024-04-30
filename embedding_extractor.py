import torch
from torchvision import models
from torchvision.models import feature_extraction
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import torchvision.transforms as transforms
import os
import glob
import pickle
from collections import defaultdict

embedding_vectors = []

model = models.get_model('efficientnet_b0')
model.eval()

feature_extractor = feature_extraction.create_feature_extractor(model, return_nodes=['avgpool'])

path = './*'
file_list = glob.glob(path)
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]

for file in file_list_jpg:
    input = cv2.imread(file)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    input = transform(input).unsqueeze(0)

    with torch.no_grad():
        embedding_vector = feature_extractor(input)

    print(embedding_vector['avgpool'].size())
    embedding_vectors.append(embedding_vector)

with open('embedding_vector.pickle', 'wb') as f:
    pickle.dump(embedding_vectors, f)
