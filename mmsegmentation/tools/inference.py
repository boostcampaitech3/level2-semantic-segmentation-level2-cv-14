import os
import os.path as osp
import argparse

import torch

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import albumentations as A
import pandas as pd
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', help='inference config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thr', type=float, default=0.001, help='rcnn nms score threshold')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='rcnn nms iou threshold')
    parser.add_argument('--work-dir', help='the dir checkpoint exists')
    parser.add_argument('--output-size', type=int, default=256, help='output size')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    seed = args.seed if args.seed else 42
    cfg.gpu_ids = [1]

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.data.test.test_mode = True
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
    
    epoch = args.checkpoint
        
    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    json_dir = './data/test.json'

    with open(json_dir, "r", encoding="utf8") as f:
        test_json = json.load(f)
    
    # sample_submisson.csv 열기
    submission = pd.DataFrame(columns=['image_id', 'PredictionString'])
    outputs = single_gpu_test(model, data_loader)

    # output_size 256
    input_size = 512
    output_size = args.output_size

    transform = A.Compose([A.Resize(output_size, output_size)])

    file_name_list = []
    preds_array = np.empty((0, output_size*output_size), dtype=np.long)

    for id, mask in enumerate(outputs):
        image = test_json['images'][id]
        file_name = image['file_name']

        # resize (256 x 256)
        temp_mask = []
        img = np.zeros((input_size,input_size,3))
        transformed = transform(image=img, mask=mask)
        mask = transformed['mask']
        temp_mask.append(mask)
        
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)
        string = oms.flatten()
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)

if __name__ == '__main__':
    main()