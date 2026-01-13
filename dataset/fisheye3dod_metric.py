from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.registry import METRICS
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes


from .fisheye3dod_eval import Fisheye3DODEval


MAP_NAME_TO_ATTR = {
    "Car": "vehicle.moving",
    "Van": "vehicle.moving",
    "Truck": "vehicle.moving",
    "Bus": "vehicle.moving",
    "Pedestrian": "pedestrian.moving",
    "Cyclist": "cycle.with_rider"
}


@METRICS.register_module()
class Fisheye3DODMetric(BaseMetric):
    def __init__(self,
                 eval_weather: Optional[List[str]] = None,
                 eval_distance: Optional[List[float]] = None,
                 ref_range: Optional[Dict[str, int]] = None,
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        super(Fisheye3DODMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.eval_weather = eval_weather
        self.eval_distance = eval_distance
        self.ref_range = ref_range
        self.backend_args = backend_args
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            result['token'] = data_sample['token']
            result['sample_idx'] = data_sample['sample_idx']
            result['gt_instances_3d'] = data_sample['eval_ann_info']
            result['pred_instances_3d'] = data_sample['pred_instances_3d']
            self.results.append(result)

    #TODO: Finish the implementation of this method with eval_weather and eval_distance
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']
        self.id_to_name = {v: k for k, v in self.dataset_meta['categories'].items()}
        pred_annos = self.format_to_nusc_annos(results, pred_flag=True)
        gt_annos = self.format_to_nusc_annos(results, pred_flag=False)
        
        eval_title = '\n================== Fisheye3DOD Evaluation ==================\n'
        fisheye3d_eval = Fisheye3DODEval(pred_annos, gt_annos, ref_range=self.ref_range)
        metrics_summary, result, details = fisheye3d_eval.main()
        result = eval_title + result
        logger.info(result)
        metric_dict = details
        # metric_dict = {
        #     'details': details,
        # }
        return metric_dict

    def split_annos_by_weather(self, annos: dict) -> Tuple[dict, dict]:
        '''
            annos: dict
                sample_token: List[dict]
                    [   
                        {
                            'sample_token': str,
                            'ego_translation': List[float],
                            'translation': List[float],
                            'size': List[float],
                            'rotation': List[float],
                            'velocity': List[float],
                            'detection_name': str,
                            'detection_score': float,
                            'attribute_name': str
                        },
                    ]
            return: dict
                weather: dict
                    sample_token: List[dict]
                        [
                            {
                                'sample_token': str,
                                 ...
                            }
                        ]
        '''
        weather_dict = defaultdict(dict)
        for sample_token, anno_list in annos.items():
            for weather in self.eval_weather:
                if weather.lower() in sample_token.lower():
                    weather_dict[weather][sample_token] = anno_list
                    break
        return weather_dict

    def split_annos_by_distance(self, annos: dict, tolerance: float) -> Tuple[dict, dict]:
        '''
        return: dict
            distance_range: dict
                sample_token: List[dict]
                    [
                        {
                            'sample_token': str,
                                ...
                        }
                    ]
        '''
        adjusted_ranges = []
        for i in range(len(self.eval_distance)):
            lower = (self.eval_distance[i - 1] if i > 0 else 0) - tolerance
            upper = self.eval_distance[i] + tolerance
            adjusted_ranges.append((lower, upper))
        
        result = defaultdict(dict)
        for sample_token, anno_list in annos.items():
            for item in anno_list:
                translation = np.array(item["translation"])
                distance = np.linalg.norm(translation)
                for i, (lower, upper) in enumerate(adjusted_ranges):
                    if lower <= distance < upper:
                        range_key = f"{lower:.1f}-{upper:.1f}"
                        if sample_token not in result[range_key]:
                            result[range_key][sample_token] = []
                        result[range_key][sample_token].append(item)
                        break
        return result

    def format_to_nusc_annos(
            self,        
            results: List[dict],
            pred_flag: bool):
        
        nusc_annos = {}
        anno_key = 'pred_instances_3d' if pred_flag else 'gt_instances_3d'
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            token = det['token']
            boxes, _ = output_to_nusc_box(det[anno_key], pred_flag=pred_flag)
            for k, box in enumerate(boxes):
                name = self.id_to_name[box.label]
                attribute_name = MAP_NAME_TO_ATTR[name]
                nusc_anno = {
                    'sample_token': token,
                    'ego_translation': box.center.tolist(),  # ego pose defined as zero
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'velocity': box.velocity[:2].tolist(),
                    'detection_name': name,
                    'detection_score': box.score,
                    'attribute_name': attribute_name,
                }
                annos.append(nusc_anno)
            nusc_annos.update({token: annos})
        return nusc_annos


def output_to_nusc_box(
        detection: dict,
        pred_flag: bool) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
        NuScenesBoxes and attribute labels.
    """
    if pred_flag:
        bbox3d = detection['bboxes_3d'].cpu()
        scores = detection['scores_3d'].numpy()
        labels = detection['labels_3d'].numpy()
    else:
        bbox3d = detection['gt_bboxes_3d'].cpu()
        labels = detection['gt_labels_3d']
        scores = np.ones(len(labels)) * -1

    attrs = None
    if 'attr_labels' in detection:
        attrs = detection['attr_labels'].numpy()
    
    # if isinstance(bbox3d, torch.Tensor) and bbox3d.device.type == 'cuda':
    #     bbox3d = bbox3d.cpu()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])

            velocity = (*bbox3d.tensor[i, 7:9],
                0.0) if bbox3d.tensor.shape[1] == 9 else (0.0, 0.0, 0.0)

            # if bbox3d.tensor.shape[1] == 7:
            #     velocity = None
            # elif bbox3d.tensor.shape[1] == 9:
            #     velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            # else:
            #     raise ValueError(
            #         'bbox3d must have 7 or 9 columns, '
            #         f'but got {bbox3d.tensor.shape[1]}')
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    elif isinstance(bbox3d, CameraInstance3DBoxes):
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        for i in range(len(bbox3d)):
            q1 = pyquaternion.Quaternion(
                axis=[0, 0, 1], radians=nus_box_yaw[i])
            q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes '
            'to standard NuScenesBoxes.')

    return box_list, attrs