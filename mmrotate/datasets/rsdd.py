from .builder import ROTATED_DATASETS
from .dota import DOTADataset
import numpy as np
import mmcv
from mmcv import print_log
from PIL import Image
import os
import torch
from mmrotate.core.evaluation import eval_rbbox_map

import os.path as osp
import xml.etree.ElementTree as ET
from mmrotate.core import poly2obb_np, obb2poly
from collections import OrderedDict

@ROTATED_DATASETS.register_module()
class RSDDDataset(DOTADataset):
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]
    def __init__(self, min_size=None, **kwargs):
        assert self.CLASSES or kwargs.get(
            'classes', None), 'CLASSES in `XMLDataset` can not be None.'
        super(RSDDDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos    

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, 'Annotations',
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        polarization = root.find('polarization').text
        resolution = root.find('resolution').text
        bboxes = []
        labels = []
        polygons = []
        bboxes_ignore = []
        labels_ignore = []
        polygons_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('robndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            # the annotation uses h as longer edge while mmcv recognize w as longer edge
            robndbox = [[float(bnd_box.find('cx').text),float(bnd_box.find('cy').text),
                       float(bnd_box.find('h').text),float(bnd_box.find('w').text),
                       float(bnd_box.find('angle').text)]]
            robndbox = torch.tensor(robndbox, dtype=torch.float32)
            bboxpoly = obb2poly(robndbox, 'le90')[0].numpy()
            try:
                x, y, w, h, a = poly2obb_np(bboxpoly, self.version)
            except:  # noqa: E722
                continue
            if difficult:
                bboxes_ignore.append([x, y, w, h, a])
                labels_ignore.append(label)
                polygons_ignore.append(bboxpoly)
            else:
                bboxes.append([x, y, w, h, a])
                labels.append(label)
                polygons.append(bboxpoly)
        if len(bboxes) == 0:
            bboxes = np.zeros((0, 5))
            labels = np.zeros((0, ))
            polygons = np.zeros((0, 8))
        else:
            bboxes = np.array(bboxes)
            labels = np.array(labels)
            polygons = np.array(polygons)
        if len(bboxes_ignore) == 0:
            bboxes_ignore = np.zeros((0, 5))
            labels_ignore = np.zeros((0, ))
            polygons_ignore = np.zeros((0, 8))
        else:
            bboxes_ignore = np.array(bboxes_ignore)
            labels_ignore = np.array(labels_ignore)
            polygons_ignore = np.array(polygons_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            polygons = polygons.astype(np.float32))
            # bboxes_ignore=bboxes_ignore.astype(np.float32),
            # labels_ignore=labels_ignore.astype(np.int64),
            # polygons_ignore = polygons_ignore.astype(np.float32))
        ann.update(polarization=polarization,
                    resolution=resolution)
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger='silent',
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.5, 0.75],
                 scale_ranges=[(0, 1e6), (0, 25), (25, 86.605), (86.605, 1e6)],
                 use_07_metric=True,
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
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'details']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        if metric == 'mAP':
            if isinstance(iou_thr, list):
                for iou in iou_thr:
                    mean_ap, _ = eval_rbbox_map(
                        results,
                        annotations,
                        scale_ranges=None,
                        iou_thr=iou,
                        dataset=self.CLASSES,
                        use_07_metric=use_07_metric,
                        logger=logger,
                        nproc=nproc)
                    eval_results['mAP_{:.2}'.format(iou)] = round(mean_ap, 3)                  
            else:
                assert isinstance(iou_thr, float)
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    use_07_metric=use_07_metric,
                    logger=logger,
                    nproc=nproc)
                eval_results['mAP'] = mean_ap
        elif metric == 'details':
            eps = np.finfo(np.float32).eps
            iou_thrs = [0.5+0.05*i for i in range(10)]
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, raw_results = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                f_measure_list = []
                for class_result in raw_results:
                    precisions = class_result['precision']
                    recalls = class_result['recall']
                    top = recalls * precisions
                    down = np.maximum(recalls + precisions,eps)
                    f_measure = np.max(2*(top/down))
                    f_measure_list.append(f_measure)
                f_score = np.mean(np.array(f_measure_list))
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 6)
                eval_results[f'F1@{int(iou_thr * 100):02d}'] = round(f_score, 6)
            # calculate aps, apm, apl at 0.5  
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=0.5,
                dataset=self.CLASSES,
                use_07_metric=False,
                logger=logger,
                nproc=nproc)    
            eval_results['mAP@.50'] = round(mean_ap[0], 6)  
            eval_results['mAP_s@.50'] = round(mean_ap[1], 6)
            eval_results['mAP_m@.50'] = round(mean_ap[2], 6)
            eval_results['mAP_l@.50'] = round(mean_ap[3], 6)      
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=0.75,
                dataset=self.CLASSES,
                use_07_metric=False,
                logger=logger,
                nproc=nproc)    
            eval_results['mAP@.75'] = round(mean_ap[0], 6)  
            eval_results['mAP_s@.75'] = round(mean_ap[1], 6)
            eval_results['mAP_m@.75'] = round(mean_ap[2], 6) 
            eval_results['mAP_l@.75'] = round(mean_ap[3], 6)    
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)        
        else:
            raise NotImplementedError

        return eval_results