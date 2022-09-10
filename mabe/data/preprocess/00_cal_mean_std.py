import numpy as np

data_path = "/cache/submission_keypoints.npy"
user_train = np.load(data_path, allow_pickle=True).item()
seqs = user_train["sequences"]
keypoints = []
for id in seqs:
    keypoint = seqs[id]["keypoints"]  # (b, num_animals, num_keypoints, 2)
    b, c, h, w = keypoint.shape
    keypoint = keypoint.reshape(b * c, h * w)
    keypoints.append(keypoint)
keypoints = np.concatenate(keypoints, axis=0)
print(1, keypoints.shape)
print(2, list(np.nanmean(keypoints, axis=0)))
print(3, list(np.nanstd(keypoints, axis=0)))
