from torch.utils.data import DataLoader
from tqdm import tqdm

from mabe.data import create_dataset
from mabe.utils import parse

opt = parse("options/video/seed_0.yml")
dataset_opt = opt["datasets"]["train"]
# dataset_opt = opt["datasets"]["test"]
train_set = create_dataset(dataset_opt)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
# train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
for idx, data in tqdm(enumerate(train_loader)):
    # print(idx)
    # for k, v in data.items():
    #     print(k)
    # break
    pass
