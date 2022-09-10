import numpy as np
from tqdm import tqdm

data_path = "/cache/frame_number_map.npy"
frame_number_map = np.load(data_path, allow_pickle=True).item()
id_list = list(frame_number_map.keys())
n = len(id_list)
print(n)
f = open(f"mouse_meta_info_test.txt", "w")
i = 0
for id in tqdm(id_list):
    f.write(f"{id}\n")
    print(frame_number_map[id])
    (st, ed) = frame_number_map[id]
    if st != i:
        assert 0
    i += 1800
f.close()
