import os

import numpy as np


def check_video_num(video_dir, name_list):
    file_list = os.listdir(video_dir)
    print(len(file_list), len(set(file_list)))
    print(len(name_list), len(set(name_list)))
    print(
        f"{len(set(file_list))} - {len(set(name_list))} = {set(name_list) - set(file_list)}"
    )


data_root = "../dataset/mouse"
submission_keypoints = np.load(
    f"{data_root}/submission_keypoints.npy", allow_pickle=True
).item()
frame_number_map = np.load(
    f"{data_root}/frame_number_map.npy", allow_pickle=True
).item()
user_train = np.load(f"{data_root}/user_train.npy", allow_pickle=True).item()

key_list = list(submission_keypoints["sequences"].keys())
name_list = [key + ".avi" for key in key_list]
check_video_num(f"{data_root}/submission_videos", name_list)
check_video_num(f"{data_root}/submission_videos_resized_224", name_list)

key_list = list(user_train["sequences"].keys())
name_list = [key + ".avi" for key in key_list]
check_video_num(f"{data_root}/userTrain_videos", name_list)
check_video_num(f"{data_root}/userTrain_videos_resized_224", name_list)

"""
1830 1830
1830 1830
1830 - 1830 = set()
1829 1829
1830 1830
1829 - 1830 = {'R6R4SNHGFM0ZOOH9A1TN.avi'}
784 784
784 784
784 - 784 = set()
784 784
784 784
784 - 784 = set()
"""
