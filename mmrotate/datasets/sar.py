# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .dota import DOTADataset
from mmrotate.core import eval_rbbox_map
import os
@ROTATED_DATASETS.register_module()
class SARDataset(DOTADataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.5, 0.75],
                 scale_ranges=[(0, 1e6), (0, 32), (32, 96), (85, 1e6)],
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
        eval_results = {}
        if metric == 'mAP':
            if isinstance(iou_thr, list):
                for iou in iou_thr:
                    mean_ap, _ = eval_rbbox_map(
                        results,
                        annotations,
                        scale_ranges=None,
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
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                eval_results['mAP'] = mean_ap
        elif metric == 'details':
            iou_thrs = [0.5+0.05*i for i in range(10)]
            ap = []
            for iou in iou_thrs:
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou,
                    dataset=self.CLASSES,
                    logger='silent',
                    nproc=nproc)    
                ap.append(mean_ap)
            mAP = sum(ap) / 10.
            eval_results['mAP'] = mAP       
            # calculate aps, apm, apl at 0.5  
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=0.5,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)    
            ap.append(mean_ap)
            eval_results['mAP@.50'] = mean_ap[0]  
            eval_results['mAP_s@.50'] = mean_ap[1]
            eval_results['mAP_m@.50'] = mean_ap[2] 
            eval_results['mAP_l@.50'] = mean_ap[3]      
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=0.75,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)    
            ap.append(mean_ap)
            eval_results['mAP@.75'] = mean_ap[0]  
            eval_results['mAP_s@.75'] = mean_ap[1]
            eval_results['mAP_m@.75'] = mean_ap[2] 
            eval_results['mAP_l@.75'] = mean_ap[3]          
        else:
            raise NotImplementedError

        return eval_results