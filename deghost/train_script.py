import subprocess



command = 'python train.py --dataset_dir "/workspace/" --logdir "./runs/toy_2/" --is_downscale --keep_query'# --resume "./runs/toy_1/val_latest_checkpoint.pth"'
res = subprocess.run(command, shell=True, capture_output=True)