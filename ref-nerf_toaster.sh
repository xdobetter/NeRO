CUDA_VISIBLE_DEVICES=7 python run_training.py --cfg configs/shape/nerf/toaster.yaml
CUDA_VISIBLE_DEVICES=7 python extract_mesh.py --cfg configs/shape/nerf/toaster.yaml
CUDA_VISIBLE_DEVICES=7 python run_training.py --cfg configs/material/nerf/toaster.yaml
CUDA_VISIBLE_DEVICES=7 python extract_materials.py --cfg configs/material/nerf/toaster.yaml
