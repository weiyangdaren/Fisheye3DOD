'''
refer from nuscenes.eval.detection.evaluate.py
'''

import numpy as np
import time
from typing import List, Dict, Tuple

from nuscenes.eval.common.loaders import _get_box_class_field
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp


DETECTION_NAMES = ["Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist"]
ATTRIBUTE_NAMES = ['pedestrian.moving', 'cycle.with_rider', 'vehicle.moving']
TP_METRICS = ['trans_err', 'scale_err', 'orient_err']
EVAL_CONFIG = {
    'class_range': {
        'Car': 48,
        'Van': 48,
        'Truck': 48,
        'Bus': 48,
        'Pedestrian': 36,
        'Cyclist': 42,
    },
    "dist_fcn": "center_distance",
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "max_boxes_per_sample": 50,
    "mean_ap_weight": 3.0
}


class FisheyeDetectionConfig(DetectionConfig):
    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int):

        assert set(class_range.keys()) == set(
            DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.class_names = self.class_range.keys()


class FisheyeDetectionMetrics(DetectionMetrics):
    def __init__(self, cfg: DetectionConfig):
        super().__init__(cfg)

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(
                    detection_name, metric_name))
            errors[metric_name] = float(np.nanmean(class_errors))
        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:
            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]
            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)
            scores[metric_name] = score
        return scores


class FisheyeDetectionBox(DetectionBox):
    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 # Translation to ego vehicle in meters.
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),
                 # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 num_pts: int = -1,
                 # The class name used in the detection challenge.
                 detection_name: str = 'car',
                 # GT samples do not have a score.
                 detection_score: float = -1.0,
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size,
                         rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name
        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(
            detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)
                          ), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name


class Fisheye3DODEval:
    def __init__(self, pred_annos, gt_annos, ref_range=None):
        eval_config = EVAL_CONFIG.copy()
        if ref_range is not None:
            class_range = {
                k: v/48 * ref_range for k, v in eval_config['class_range'].items()}
            eval_config['class_range'] = class_range
        self.cfg = FisheyeDetectionConfig.deserialize(eval_config)
        self.pred_boxes = EvalBoxes.deserialize(
            pred_annos, FisheyeDetectionBox)
        self.gt_boxes = EvalBoxes.deserialize(
            gt_annos, FisheyeDetectionBox)
        self.pred_boxes = self.filter_eval_boxes(
            self.pred_boxes, self.cfg.class_range)
        self.gt_boxes = self.filter_eval_boxes(
            self.gt_boxes, self.cfg.class_range)
        self.sample_tokens = self.gt_boxes.sample_tokens

    def filter_eval_boxes(self, eval_boxes, max_dist):
        class_field = _get_box_class_field(eval_boxes)
        for ind, sample_token in enumerate(eval_boxes.sample_tokens):
            # Filter on distance
            eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                              box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        return eval_boxes

    def evaluate(self):
        start_time = time.time()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes,
                                class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        metrics = FisheyeDetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall,
                             self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(
                    class_name, self.cfg.dist_th_tp)]
                tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)
        return metrics, metric_data_list

    def main(self):
        metrics, metric_data_list = self.evaluate()
        metrics_summary = metrics.serialize()

        details = {}
        # Print high-level metrics.
        result = '{:<12s} {:<6.4f}\n'.format(
            'mAP:', metrics_summary['mean_ap'])
        details['mAP'] = metrics_summary['mean_ap']
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            result += '{:<12s} {:<6.4f}\n'.format(
                err_name_mapping[tp_name] + ':', tp_val)
            details[err_name_mapping[tp_name]] = tp_val
        result += '{:<12s} {:<6.4f}\n'.format('NDS:',
                                              metrics_summary['nd_score'])
        result += '{:<12s} {:<6.3f}s\n'.format('Eval time:',
                                               metrics_summary['eval_time'])
        details['NDS'] = metrics_summary['nd_score']

        # Print per-class metrics.
        # result += '----------------Per-class results----------------\n'
        result += '{:<20s}\t{:<6s}\t{:<6s}\t{:<6s}\t{:<6s}'.format(
            'Object Class', 'AP', 'ATE', 'ASE', 'AOE')
        for dist_th in self.cfg.dist_ths:
            result += '\t{:<6s}'.format('AP@{}'.format(dist_th))
        result += '\n'

        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        label_aps = metrics_summary['label_aps']
        for class_name in class_aps.keys():
            result += '{:<20s}\t{:<6.4f}\t{:<6.4f}\t{:<6.4f}\t{:<6.4f}'.format(
                class_name,
                class_aps[class_name],
                class_tps[class_name]['trans_err'],
                class_tps[class_name]['scale_err'],
                class_tps[class_name]['orient_err']
            )
            details[f"{class_name}_AP"] = class_aps[class_name]
            details[f"{class_name}_ATE"] = class_tps[class_name]['trans_err']
            details[f"{class_name}_ASE"] = class_tps[class_name]['scale_err']
            details[f"{class_name}_AOE"] = class_tps[class_name]['orient_err']

            for dist_th in self.cfg.dist_ths:
                result += '\t{:<6.4f}'.format(label_aps[class_name][dist_th])
                details[f"{class_name}_AP@{dist_th}"] = label_aps[class_name][dist_th]
            result += '\n'
        return metrics_summary, result, details
