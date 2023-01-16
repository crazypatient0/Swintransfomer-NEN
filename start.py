import os

# cmd = 'python main.py --cfg configs/swinv2/swinv2_tiny_patch4_window16_256.yaml --local_rank 0 --batch-size 8 --accumulation-steps 8'
# cmd = 'python main.py --cfg configs/swinv2/swinv2_base_patch4_window16_256.yaml --local_rank 0 --batch-size 4 --accumulation-steps 16 '
#
# os.system(cmd)


# cmd3 = 'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
# --cfg configs/swinv2/swinv2_tiny_patch4_window16_256.yaml  --data-path dataset/pddd/'
#
# os.system(cmd)
#
# cmd2 = 'tensorboard --logdir=D:\AI\swin-transformer\Swin-Transformer'
#

# cmd4 = 'python pridiction.py  --cfg configs/swinv2/swinv2_tiny_patch4_window8_256.yaml   \
#             --ckp_path predictpath/0926-ep457-rbg-92.93.pth --img_path dataset/pd2/'
#
# os.system(cmd4)
#
# cmd4 = 'python pridiction.py  --cfg configs/swinv2/swinv2_base_patch4_window16_256.yaml   \
#             --ckp_path predictpath/1029-ep128-rgb-82-.pth --img_path dataset/pd2/'
#
# os.system(cmd4)

# heat_map = 'python heat_map.py --cfg configs/swinv2/swinv2_tiny_patch4_window16_256-512.yaml --data-path dataset/11/test4.jpg --pretrained predictpath/1021-ep141-gry-86.69.pth --local_rank 0 '
# os.system(heat_map)

img9 = 'python img_pre_9.py --ids 11111111 --imgpath 00000.jpg'
os.system(img9)