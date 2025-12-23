from typing import Optional, Sequence

import mmengine
import numpy as np
from pathlib import Path
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class SaveDetectionHook(Hook):
    def __init__(self,
                 score_thr: float = None,
                 class_names: Optional[list] = None,
    ) -> None:
        self.score_thr = score_thr
        if class_names is not None:
            self.label2name = {i: name for i, name in enumerate(class_names)}
        else:
            self.label2name = None
 
    def init_store(self):
        self.detection_outputs = []
    
    def save_results(self, results):
        for result in results:
            save_result = {}
            save_result['token'] = result.metainfo['token']
            save_result['annos'] = {}
            labels_3d = result.pred_instances_3d.labels_3d
            bboxes_3d = result.pred_instances_3d.bboxes_3d
            scores_3d = result.pred_instances_3d.scores_3d
            if self.score_thr is not None:
                inds = scores_3d > self.score_thr
                labels_3d = labels_3d[inds]
                bboxes_3d = bboxes_3d[inds]
                scores_3d = scores_3d[inds]

            if self.label2name is not None:
                labels_3d = np.array([self.label2name[l.item()] for l in labels_3d])
            else:
                labels_3d = labels_3d.numpy()
            save_result['annos']['pred_names'] = labels_3d
            save_result['annos']['pred_boxes'] = bboxes_3d.tensor.cpu().numpy()
            save_result['annos']['pred_scores'] = scores_3d.cpu().numpy()
            self.detection_outputs.append(save_result)
    
    def dump_results(self, runner: Runner):
        epoch = runner.epoch
        save_path = Path(runner.log_dir) / 'save_detection' / f'epoch_{epoch}.pkl'
        dump_data = {
            'epoch': epoch,
            'data_list': self.detection_outputs
        }
        mmengine.dump(dump_data, save_path)
        runner.logger.info(f'Saved {len(self.detection_outputs)} detections to {save_path}')
            
    def before_val_epoch(self, runner: Runner):
        self.init_store()
    
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[dict]) -> None:
        self.save_results(outputs)

    def after_val_epoch(self, runner: Runner, metrics: dict = None) -> None:
        self.dump_results(runner)
    
    def before_test_epoch(self, runner: Runner):
        self.init_store()
    
    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[dict]) -> None:
        self.save_results(outputs)
    
    def after_test_epoch(self, runner: Runner, metrics: dict = None) -> None:
        self.dump_results(runner)