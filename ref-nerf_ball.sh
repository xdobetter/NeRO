CUDA_VISIBLE_DEVICES=2 python run_training.py --cfg configs/shape/nerf/ball.yaml
CUDA_VISIBLE_DEVICES=2 python extract_mesh.py --cfg configs/shape/nerf/ball.yaml
CUDA_VISIBLE_DEVICES=2 python run_training.py --cfg configs/material/nerf/ball.yaml
CUDA_VISIBLE_DEVICES=2 python extract_materials.py --cfg configs/material/nerf/ball.yaml
