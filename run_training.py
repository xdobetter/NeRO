# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' #卡7存在问题
import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
flags = parser.parse_args()

Trainer(load_cfg(flags.cfg)).run()
