import torch
import pickle
import math
import cv2
import os

with open('dinov2_embedding.pickle', 'rb') as fw:
    total_features = pickle.load(fw)

total_features_reshape = torch.reshape(total_features, (total_features.shape[0], total_features.shape[2], int(math.sqrt(total_features.shape[1])), -1)) #(image 개수, 1024, 70, 70)

# 주변 프레임과 feature 값 비교해서 값이 크게 차이 나는 프레임 추출
diff = torch.zeros_like(total_features_reshape)

diff[0] = torch.abs(total_features_reshape[1] - total_features_reshape[0]) #첫번째 이미지

diff[-1] = torch.abs(total_features_reshape[-1] - total_features_reshape[-2]) #마지막 이미지

for i in range(1, total_features_reshape.size(0) - 1): #나머지 이미지
    prev_diff = torch.abs(total_features_reshape[i] - total_features_reshape[i - 1]) #이전 프레임과 비교
    next_diff = torch.abs(total_features_reshape[i + 1] - total_features_reshape[i]) #이후 프레임과 비교
    diff[i] = prev_diff + next_diff

# 주변 프레임과의 차이를 다 더함
diff_sum = []
for f in diff:
    diff_sum.append(f.sum().item())

keyframe_candi = []
threshold = 4000000

for i, s in enumerate(diff_sum):
    if s > threshold:
        keyframe_candi.append(i+1)
        print(i+1, diff_sum[i])

#핵심 키프레임으로 영상 만들기
folder_path = "./keyframes/"
image_paths = sorted(os.listdir(folder_path))
important_frames = []

for index in keyframe_candi:
    img = cv2.imread(os.path.join(folder_path, image_paths[index-1]))
    height, width, channel = img.shape
    size = (width, height)
    important_frames.append(img)
out = cv2.VideoWriter('./dinov2_keyframes.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(important_frames)):
    out.write(important_frames[i])

out.release()