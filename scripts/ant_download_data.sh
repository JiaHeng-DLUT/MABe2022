mkdir ../dataset
mkdir ../dataset/ants
cd ../dataset/ants

wget -O user_train.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/user_train.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=55aa68532764949f2f094101de67f4ca99e6a4d73d0143f41092a14f7f023330"
wget -O userTrain_videos_resized_224.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/userTrain_videos_resized_224.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=92ef03c85322b5efea494b0227436dce4b731905e26875bb90928b635832134d"
wget -O userTrain_videos.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/userTrain_videos.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=abe3814d895ee2b2e738b8e5cd3ceeaeb2478585729221e013aa52ff6d58b34c"
wget -O submission_videos_resized_224.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/submission_videos_resized_224.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d7aa194a425918f8c6016be254daaf3500f1c77d574e0af801fa03a93e8bf693"
wget -O submission_videos.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/submission_videos.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=dc04a0070c2df085ba9e4274b4dc264e18030357eff22864cda65478d17f8719"
wget -O frame_number_map.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/frame_number_map.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8afe765658b169cacd7fb8b7e59b7291187c18e354dea2a6abf14d8935e611f8"
wget -O submission_keypoints.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/submission_keypoints.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=fd13b61673f858db984eabbc8356b52c8c8cf4ddcb112b0b7165c7d92ab8010b"
wget -O sample_submission.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/ant-beetle-round2/public/v0.1/sample_submission.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220915%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220915T181641Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c26874b80813f23869fdca1d68dd385fe46e61be498b7a6ce2ac62099c9d1999"

mkdir submission_videos
unzip -q submission_videos.zip -d submission_videos
mkdir submission_videos_resized_224
unzip -q submission_videos_resized_224.zip -d submission_videos_resized_224
mkdir userTrain_videos
unzip -q userTrain_videos.zip -d userTrain_videos
mkdir userTrain_videos_resized_224
unzip -q userTrain_videos_resized_224.zip -d userTrain_videos_resized_224
