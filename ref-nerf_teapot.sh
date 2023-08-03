CUDA_VISIBLE_DEVICES=6 python run_training.py --cfg configs/shape/nerf/teapot.yaml
CUDA_VISIBLE_DEVICES=6 python extract_mesh.py --cfg configs/shape/nerf/teapot.yaml
CUDA_VISIBLE_DEVICES=6 python run_training.py --cfg configs/material/nerf/teapot.yaml
CUDA_VISIBLE_DEVICES=6 python extract_materials.py --cfg configs/material/nerf/teapot.yaml
