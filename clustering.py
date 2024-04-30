import pickle
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

#임베딩 벡터 파일 가져오기
with open('embedding_vector.pickle', 'rb') as fw:
  load_dict = pickle.load(fw)
toarray = np.array(load_dict)
toarray = toarray.squeeze() #(1800, 1280)

#임베딩 벡터 차원 축소
tsne2 = TSNE(n_components=2)
tsne3 = TSNE(n_components=3)
embedding_2d = tsne2.fit_transform(toarray)
embedding_3d = tsne3.fit_transform(toarray)

def k_means(embedding):
  kmeans = KMeans(n_clusters=2, random_state=42)
  labels = kmeans.fit_predict(embedding)
  return labels

def dbscan(embedding, eps, min_samples):
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)
  labels = dbscan.fit_predict(embedding)
  return dbscan, labels

def make_groups(labels):
  path = './*'
  file_list = glob.glob(path)
  file_list_jpg = [file for file in file_list if file.endswith(".jpg")]

  groups = {}
  for file, cluster in zip(file_list_jpg, labels):
    if cluster not in groups.keys():
      groups[cluster] = []
      groups[cluster].append(file)
    else:
      groups[cluster].append(file)
  return groups

def make_video(groups, label, cluster):
  frame_array = []
  for file in groups.get(label):
    img = cv2.imread(file)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)
  out = cv2.VideoWriter(f"./{cluster}_{label}.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

  for i in range(len(frame_array)):
    out.write(frame_array[i])
  out.release()

def see(db):
  cmap = plt.cm.get_cmap('tab20')
  colors = [cmap(i) for i in range(len(np.unique(db)))]
  print(np.unique(db))
  for label, color in enumerate(colors):
      label = label-1
      if label == -1:
          color = 'k'

      cluster_mask = (db == label) 
      plt.scatter(embedding_2d[cluster_mask, 0], embedding_2d[cluster_mask, 1], color=color, label = f'Cluster {label}')

  plt.title('DBSCAN Clustering of Embedding Vectors')
  plt.xlabel('Dimension 1')
  plt.ylabel('Dimension 2')
  plt.legend()
  plt.show()

def make_dataframe(embed, dbscan, labels):
  cluster_df = pd.DataFrame(embed, columns=['pc1', 'pc2'])
  outlier_list = []
  for idx, label in enumerate(dbscan.labels_):
      if label == -1:
          outlier_list.append(idx)

  core_list = dbscan.core_sample_indices_.tolist()

  cluster_df.loc[outlier_list, 'cluster'] = 'outlier'
  cluster_df.loc[core_list, 'cluster'] = 'core'
  cluster_df['cluster'] = cluster_df['cluster'].fillna('border')

  cluster_df['cluster_i'] = labels

 # 각 군집에서 밀도가 가장 높은 포인트를 저장할 데이터프레임 생성 
def extract_density_peak_points(dbscan_df):
    density_peak_points = pd.DataFrame(columns=dbscan_df.columns)
    num_clusters = dbscan_df['cluster_i'].max() + 1
    
    for cluster_id in range(num_clusters):
        cluster_data = dbscan_df[dbscan_df['cluster_i'] == cluster_id]
        core_points = cluster_data[cluster_data['cluster'] == 'core']
      
        if core_points.empty:
            continue
        
        distances = pairwise_distances(core_points[['pc1', 'pc2']], metric='euclidean')
        
        # 거리 중심으로부터의 거리가 최소가 되는 포인트 선택
        min_distance_index = np.argmin(np.sum(distances, axis=1))
        density_peak_point = core_points.iloc[min_distance_index]
        
        # 밀도가 가장 높은 포인트를 결과 데이터프레임에 추가
        density_peak_points = density_peak_points.append(density_peak_point)
    
    return density_peak_points