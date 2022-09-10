import numpy as np
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.decomposition import PCA

pca_feat_list = [
    "../experiments/video/seed_2/models/net_2000_test_feats.npy",
    "../experiments/keypoint/seed_2/models/net_1300_test_feats.npy",
    "../experiments/pointnet.npy",
]
print(0, pca_feat_list)
assert len(pca_feat_list) != 0, "feat_list is empty"
embeddings = [np.load(path, allow_pickle=True) for path in pca_feat_list]
for i in range(len(embeddings)):
    print(embeddings[i].shape)
embeddings_concat = np.concatenate(embeddings, axis=1)
print(1, embeddings_concat.shape)
pca = PCA(n_components=128)
embeddings_pca = pca.fit_transform(embeddings_concat)
print(2, embeddings_pca.shape)

feature = embeddings_pca
np.save("seed_2.npy", feature)
