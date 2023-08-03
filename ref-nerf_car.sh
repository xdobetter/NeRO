CUDA_VISIBLE_DEVICES=3 python run_training.py --cfg configs/shape/nerf/car.yaml
CUDA_VISIBLE_DEVICES=3 python extract_mesh.py --cfg configs/shape/nerf/car.yaml
CUDA_VISIBLE_DEVICES=3 python run_training.py --cfg configs/material/nerf/car.yaml
CUDA_VISIBLE_DEVICES=3 python extract_materials.py --cfg configs/material/nerf/car.yaml
