CUDA_VISIBLE_DEVICES=6 python run_training.py --cfg configs/shape/dtu/dtu_scan97.yaml
CUDA_VISIBLE_DEVICES=6 python extract_mesh.py --cfg configs/shape/dtu/dtu_scan97.yaml
CUDA_VISIBLE_DEVICES=6 python run_training.py --cfg configs/material/dtu/dtu_scan97.yaml
CUDA_VISIBLE_DEVICES=6 python extract_materials.py --cfg configs/material/dtu/dtu_scan97.yaml
