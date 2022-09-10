import random

import numpy as np

seed = 0
random.seed(seed)


data_path = "/cache/submission_keypoints.npy"
user_train = np.load(data_path, allow_pickle=True).item()
seqs = user_train["sequences"]
id_list = list(seqs.keys())
n = len(id_list)
print(n)
f = open(f"mouse_meta_info_train_{seed}.txt", "w")
for id in id_list:
    f.write(f"{id}\n")
f.close()


data_path = "/cache/user_train.npy"
user_train = np.load(data_path, allow_pickle=True).item()
seqs = user_train["sequences"]
id_list = list(seqs.keys())
n = len(id_list)
print(n)
f = open(f"mouse_meta_info_val_{seed}.txt", "w")
for id in id_list:
    f.write(f"{id}\n")
f.close()
