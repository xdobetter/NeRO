CUDA_VISIBLE_DEVICES=5 python run_training.py --cfg configs/shape/nerf/helmet.yaml
CUDA_VISIBLE_DEVICES=5 python extract_mesh.py --cfg configs/shape/nerf/helmet.yaml
CUDA_VISIBLE_DEVICES=5 python run_training.py --cfg configs/material/nerf/helmet.yaml
CUDA_VISIBLE_DEVICES=5 python extract_materials.py --cfg configs/material/nerf/helmet.yaml
