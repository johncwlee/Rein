import copy
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple
import random

import mmengine
import mmcv
import mmengine.fileio as fileio
import torch
import numpy as np
from mmengine.dataset import Compose
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from mmcv.transforms.utils import cache_randomness

from mmseg.datasets import CityscapesDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class CityscapesCOCODataset(CityscapesDataset):
    """Cityscapes with COCO outliers dataset."""

    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle', 'anomaly'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
                 [115, 5, 115]])

@TRANSFORMS.register_module()
class MixCOCO(BaseTransform):
    def __init__(self, coco_root: str, ood_idx: int = 254):
        self.coco_root = coco_root
        files = list(fileio.list_dir_or_file(
            dir_path=osp.join(coco_root, "ood_seg_train2017"),
            list_dir=False,
            suffix=".png",
            recursive=True,
            backend_args=None
        ))
        targets = [osp.join(coco_root, "ood_seg_train2017", f) for f in files]
        images = [t.replace("ood_seg_", "").replace("png", "jpg") for t in targets]
        self.coco_data = list(zip(images, targets))
        self.ood_idx = ood_idx
    
    @cache_randomness
    def _random_select(self) -> Tuple[int, int]:
        idx = np.random.randint(len(self.coco_data))
        coco_data = self.coco_data[idx]
        return coco_data

    @staticmethod
    def extract_bboxes(mask: np.ndarray) -> np.ndarray:
        """Extract bounding boxes from the mask."""
        mask = torch.from_numpy(mask)
        if mask.sum() == 0:
            return np.zeros(4, dtype=np.int32)

        hori_indices = torch.where(torch.any(mask,dim=0))[0]
        vert_indices = torch.where(torch.any(mask,dim=1))[0]
        x1, x2 = hori_indices[[0, -1]]
        y1, y2 = vert_indices[[0, -1]]
        box = torch.tensor([y1, x1, y2+1, x2+1], dtype=torch.int32)
        return box.numpy()

    def transform(self, results: dict) -> dict:
        coco_data = self._random_select()
        coco_img_path, coco_ood_tgt_path = coco_data
        results["coco_img_path"] = coco_img_path
        results["coco_ood_tgt_path"] = coco_ood_tgt_path

        img_bytes = fileio.get(coco_img_path, backend_args=None)
        coco_img = mmcv.imfrombytes(
            img_bytes, flag='color', backend='cv2')
        
        img_bytes = fileio.get(coco_ood_tgt_path, backend_args=None)
        coco_tgt = mmcv.imfrombytes(
            img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        
        train_id_out = self.ood_idx
        mask = coco_tgt == train_id_out
        box = self.extract_bboxes(mask)
        y1, x1, y2, x2 = box

        coco_tgt_cut = coco_tgt[y1:y2, x1:x2]
        coco_img_cut = coco_img[y1:y2, x1:x2, :]

        #? Pick random location to insert COCO image
        h_start = torch.randint(0, results['img_shape'][0] - coco_tgt_cut.shape[0] + 1, (1,)).item()
        h_end = h_start + coco_tgt_cut.shape[0]
        w_start = torch.randint(0, results['img_shape'][1] - coco_tgt_cut.shape[1] + 1, (1,)).item()
        w_end = w_start + coco_tgt_cut.shape[1]

        #? Insert COCO image into Cityscapes image
        results['img'][h_start:h_end, w_start:w_end][coco_tgt_cut == train_id_out] = \
            coco_img_cut[coco_tgt_cut == train_id_out]
        results['gt_seg_map'][h_start:h_end, w_start:w_end][coco_tgt_cut == train_id_out] = 19
        
        return results

    def __repr__(self):
        return self.__class__.__name__