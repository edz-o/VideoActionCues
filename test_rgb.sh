# Test RGB only model
CUDA_VISIBLE_DEVICES=0 python tools/test_recognizer.py configs/nturgbd/i3d_kinetics400_3d_rgb_inception.py modelzoo/epoch_121_rgb.pth --gpus 1
# Test Synthetic data augmentated RGB model
CUDA_VISIBLE_DEVICES=0 python tools/test_recognizer.py configs/nturgbd/i3d_kinetics400_3d_rgb_inception.py modelzoo/epoch_30_sim_augmentation.pth --gpus 1
