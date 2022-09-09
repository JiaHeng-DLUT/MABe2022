mkdir ../dataset
mkdir ../dataset/mouse
cd ../dataset/mouse

wget -O frame_number_map.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/frame_number_map.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=bd401a8ec07ccf5800cf1218846efcb08bcaa5b0fe9350643b4a98db8c47078b"
wget -O sample_submission.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/sample_submission.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0b16f81edf2af2dd7b0936bf8f8868564d219f6ccef9c0d8ea420ec633a7fe2b"
wget -O submission_keypoints.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/submission_keypoints.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f7880c28e416d4b68e67819679bbd97885e0726f6337d4dc2b16b3f5ebbfdf7f"
wget -O submission_videos.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/submission_videos.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a4553b07ea4d51f1e06b0dc9306709cc1fae35eec07af72117d20e80a23fa966"
wget -O submission_videos_resized_224.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/submission_videos_resized_224.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=bb7faa661f7e66a74c384cb1400f6254766b663f58468e7c470d86424b7ade8b"
wget -O userTrain_videos.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/userTrain_videos.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0119b6ec4b937e444ca191eeb8a0476d4de22323551c2e1b86dc33d8ddbfc418"
wget -O userTrain_videos_resized_224.zip "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/userTrain_videos_resized_224.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=99b176a330b23662b3d672dd870ce5412022378906d250187320e76ce9def551"
wget -O user_train.npy "https://aicrowd-private-datasets.s3.us-west-002.backblazeb2.com/mabe-2022/mouse-triplets-round2/public/v0.1/user_train.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000019%2F20220909%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20220909T162622Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=633e3e96946d25300ed479525de1cf0f39fc0d8f9e82806586b80c41a28ebffd"

mkdir submission_videos
unzip -q submission_videos.zip -d submission_videos
mkdir submission_videos_resized_224
unzip -q submission_videos_resized_224.zip -d submission_videos_resized_224
mkdir userTrain_videos
unzip -q userTrain_videos.zip -d userTrain_videos
mkdir userTrain_videos_resized_224
unzip -q userTrain_videos_resized_224.zip -d userTrain_videos_resized_224
