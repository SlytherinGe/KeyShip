'''
Author: SlytherinGe
LastEditTime: 2021-12-24 11:57:40
'''
import numpy as np
import cv2
from pycocotools.coco import COCO

from mmdet.datasets import CocoDataset
from .builder import ROTATED_DATASETS
import os
from mmrotate.core.evaluation import eval_rbbox_map
import mmcv

@ROTATED_DATASETS.register_module()
class HRSIDDataset(CocoDataset):

    CLASSES = ('ship',)

    def __init__(self, version='oc', *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.version = version


    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue

            gt_mask = self.coco.annToMask(ann)             
            if np.max(gt_mask)!= 0:
                contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_contour = max(contours, key=len)
                rect_box = cv2.minAreaRect(max_contour)
                poly = cv2.boxPoints(rect_box).flatten()
                x, y, w, h, a = rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1], rect_box[2]
                if w<1 or h<1:
                    print('error target!!!')
                    print('img id:', ann['image_id'])
                    continue
                while not 0 > a >= -90:
                    if a >= 0:
                        a -= 90
                        w, h = h, w
                    else:
                        a += 90
                        w, h = h, w
                a = a / 180 * np.pi
                assert 0 > a >= -np.pi / 2
            else:
                print('empty target!!!')
                continue
            if ann['iscrowd']:
                gt_bboxes_ignore.append([x, y, w, h, a])
                gt_polygons_ignore.append(poly)
                gt_labels_ignore.append(self.cat2label[ann['category_id']])
            else:
                gt_bboxes.append([x, y, w, h, a])
                gt_polygons.append(poly)
                gt_labels.append(self.cat2label[ann['category_id']])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_polygons = np.array(gt_polygons, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_polygons = np.zeros((0, 8), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            gt_polygons_ignore = np.array(gt_polygons_ignore, dtype=np.float32)
            gt_labels_ignore = np.array(gt_labels_ignore, dtype=np.int64)
        else:
            gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)
            gt_polygons_ignore = np.zeros((0, 8), dtype=np.float32)
            gt_labels_ignore = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, polygons=gt_polygons, 
            bboxes_ignore=gt_bboxes_ignore, labels_ignore=gt_labels_ignore, polygons_ignore=gt_polygons_ignore)

        return ann

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=16):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            if isinstance(iou_thr, list):
                for iou in iou_thr:
                    mean_ap, _ = eval_rbbox_map(
                        results,
                        annotations,
                        scale_ranges=scale_ranges,
                        iou_thr=iou,
                        dataset=self.CLASSES,
                        logger=logger,
                        nproc=nproc)
                    eval_results['mAP_{:.2}'.format(iou)] = mean_ap                    
            else:
                assert isinstance(iou_thr, float)
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                eval_results['mAP'] = mean_ap
        return eval_results
