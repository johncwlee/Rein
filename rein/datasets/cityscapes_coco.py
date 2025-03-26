import copy
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple
import warnings

import mmengine
import mmcv
import mmengine.fileio as fileio
from mmengine.structures import PixelData
import torch
import numpy as np
from mmcv.transforms import to_tensor
from mmengine.dataset import Compose
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from mmcv.transforms.utils import cache_randomness

from mmseg.datasets import CityscapesDataset
from mmseg.registry import DATASETS
from mmseg.structures import SegDataSample


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

@TRANSFORMS.register_module()
class MixCOCOV2(BaseTransform):
    def __init__(self, 
                 coco_root: str, 
                 mix_pipeline: List[Union[dict, Callable]] = [],
                 ood_pipeline: List[Union[dict, Callable]] = [],
                 ood_idx: int = 254):
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
        self.mix_pipeline = Compose(mix_pipeline)
        self.ood_pipeline = Compose(ood_pipeline)
    
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
        
        coco_results = {**results}
        coco_results['img'] = coco_img
        coco_results['gt_seg_map'] = coco_tgt
        coco_results['img_shape'] = coco_img.shape[:2]
        coco_results['ori_shape'] = coco_img.shape[:2]
        coco_results2 = copy.deepcopy(coco_results)

        coco_mix_out = self.mix_pipeline(coco_results)
        coco_ood_out = self.ood_pipeline(coco_results2)
        
        results['coco_img'] = coco_ood_out['img']
        coco_ood_out['gt_seg_map'][coco_ood_out['gt_seg_map'] == self.ood_idx] = 1
        results['coco_gt_seg_map'] = coco_ood_out['gt_seg_map']
        
        train_id_out = self.ood_idx
        coco_img, coco_tgt = coco_mix_out['img'], coco_mix_out['gt_seg_map']
        mask = coco_tgt == train_id_out
        box = self.extract_bboxes(mask)
        y1, x1, y2, x2 = box

        coco_tgt_cut = coco_tgt[y1:y2, x1:x2]
        coco_img_cut = coco_img[y1:y2, x1:x2, :]
        
        #? If the COCO image is too large, resize by half
        img_h, img_w = results['img_shape']
        if coco_tgt_cut.shape[0] > img_h or coco_tgt_cut.shape[1] > img_w:
            ratio = 0.5
            coco_tgt_cut = mmcv.imresize(coco_tgt_cut, 
                                         (int(coco_img_cut.shape[0] * ratio), int(coco_img_cut.shape[1] * ratio)), 
                                         interpolation='nearest')
            coco_img_cut = mmcv.imresize(coco_img_cut, 
                                         (int(coco_img_cut.shape[0] * ratio), int(coco_img_cut.shape[1] * ratio)), 
                                         interpolation='bilinear')

        #? Pick random location to insert COCO image
        try:
            h_start = torch.randint(0, img_h - coco_tgt_cut.shape[0] + 1, (1,)).item()
        except:
            print(box)
            print(np.unique(coco_tgt))
            print(coco_tgt_cut.shape, results['img_shape'])
            raise
        h_end = h_start + coco_tgt_cut.shape[0]
        w_start = torch.randint(0, img_w - coco_tgt_cut.shape[1] + 1, (1,)).item()
        w_end = w_start + coco_tgt_cut.shape[1]

        #? Insert COCO image into Cityscapes image
        results['img'][h_start:h_end, w_start:w_end][coco_tgt_cut == train_id_out] = \
            coco_img_cut[coco_tgt_cut == train_id_out]
        results['gt_seg_map'][h_start:h_end, w_start:w_end][coco_tgt_cut == train_id_out] = 19
        
        return results

    def __repr__(self):
        return self.__class__.__name__

@TRANSFORMS.register_module()
class RandomChoiceResizeV2(BaseTransform):
    def __init__(
        self,
        scale_factors: Sequence[float],
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:
        super().__init__()
        self.scale_factors = scale_factors
        assert mmengine.is_seq_of(self.scale_factors, float)

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Resize object
        self.resize = TRANSFORMS.build({'scale_factor': (0.0, 0.0), **self.resize_cfg})

    def _random_select(self) -> Tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """
        scale_factor_idx = np.random.randint(len(self.scale_factors))
        scale_factor = self.scale_factors[scale_factor_idx]
        return scale_factor, scale_factor_idx

    def transform(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """
        target_scale_factor, scale_factor_idx = self._random_select()
        self.resize.scale_factor = (target_scale_factor, target_scale_factor)
        results = self.resize(results)
        results['scale_factor_idx_v2'] = scale_factor_idx
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale_factors={self.scale_factors}'
        repr_str += f', resize_cfg={self.resize_cfg})'
        return repr_str

@TRANSFORMS.register_module()
class PackSegInputsWithCOCO(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img
        if 'coco_img' in results:
            img = results['coco_img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['coco_inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        if 'coco_gt_seg_map' in results:
            if len(results['coco_gt_seg_map'].shape) == 2:
                data = to_tensor(results['coco_gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth coco '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["coco_gt_seg_map"].shape}')
                data = to_tensor(results['coco_gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.coco_gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str