import cv2
import os

folder_path = "./keyframes/"
image_paths = sorted(os.listdir(folder_path))
keyframes = []
for image in image_paths:
    img = cv2.imread(os.path.join(folder_path, image))
    height, width, channel = img.shape
    size = (width, height)
    keyframes.append(img)

out = cv2.VideoWriter('./keyframes.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(keyframes)):
    out.write(keyframes[i])

out.release()