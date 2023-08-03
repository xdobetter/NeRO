CUDA_VISIBLE_DEVICES=4 python run_training.py --cfg configs/shape/nerf/coffee.yaml
CUDA_VISIBLE_DEVICES=4 python extract_mesh.py --cfg configs/shape/nerf/coffee.yaml
CUDA_VISIBLE_DEVICES=4 python run_training.py --cfg configs/material/nerf/coffee.yaml
CUDA_VISIBLE_DEVICES=4 python extract_materials.py --cfg configs/material/nerf/coffee.yaml
