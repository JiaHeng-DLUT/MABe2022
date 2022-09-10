import random

import numpy as np

random.seed(0)

num_video = 784
num_frame = 1800

index_list = []
for i in range(9):
    index = list(range(num_video * num_frame))
    random.shuffle(index)
    index_list.append(index)
np.savetxt("../data/mouse/meta_info_val_0.txt", np.array(index_list), fmt="%s")
