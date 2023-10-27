CUDA_VISIBLE_DEVICES=0 python run_training.py --cfg configs/shape/nerf/lego.yaml
CUDA_VISIBLE_DEVICES=0 python extract_mesh.py --cfg configs/shape/nerf/lego.yaml
CUDA_VISIBLE_DEVICES=0 python run_training.py --cfg configs/material/nerf/lego.yaml
CUDA_VISIBLE_DEVICES=0 python extract_materials.py --cfg configs/material/nerf/lego.yaml
