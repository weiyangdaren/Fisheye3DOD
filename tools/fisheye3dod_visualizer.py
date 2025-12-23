# Lyft dataset SDK
# based on the code written by Alex Lang and Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

from pathlib import Path
from typing import Any, List, Tuple, Union

import argparse
import matplotlib.pyplot as plt
import numpy as np
import mmengine
from tqdm import tqdm
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from collections import defaultdict


from ..utils.lyft_sdk import Box, LidarPointCloud
from ..utils.lyft_sdk import view_points



def get_color(category_name: str) -> Tuple[int, int, int]:
    """Provides the default colors based on the category names.
    This method works for the general Lyft Dataset categories, as well as the Lyft Dataset detection categories.

    Args:
        category_name:

    Returns:

    """
    # if category_name in ['Car', 'Van', 'Truck', 'Bus']:
    #     return 255, 158, 0  # Orange
    # elif category_name in ["Cyclist"]:
    #     return 255, 61, 99  # Red
    # elif category_name in "Pedestrian":
    #     return 0, 0, 230  # Blue
    # elif "cone" in category_name or "barrier" in category_name:
    #     return 0, 0, 0  # Black
    # else:
    #     return 255, 0, 255  # Magenta

    def hex_to_rgb(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    
    color_map = {
        'Car':         (255,  50,  50),    #  (#FF3232)
        'Van':         (255, 127,  14),    #  (#ff7f0e)
        'Truck':       (  0,  82, 255),    #  (#0052FF)
        'Bus':         ( 28, 173, 228),    #  (#1CADE4)
        'Pedestrian':  (151, 100,  75),    #  (#97644B)
        'Cyclist':     (147, 112, 219),    #  (#9370DB)
    }
    return color_map.get(category_name, (255, 0, 255))

def rotation_points_and_boxes_by_z_axis(points, boxes, angle: float) -> np.ndarray:
    """Rotate points and boxes around the Z axis (up-axis) by a given angle.
    Args:
        points: Points to rotate.
        boxes: Boxes to rotate.
        angle: Angle in radians.

    Returns: Rotated points and boxes.

    """
    c = np.cos(angle)
    s = np.sin(angle)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    quan_z = Quaternion(axis=(0, 0, 1), angle=angle)

    print(points.shape)

    # Rotate points.
    rotated_points = np.dot(points, rot_z.T).T

    # Rotate boxes.
    for box in boxes:
        box.rotate_around_origin(quan_z)
    return rotated_points, boxes


class Fisheye3DODVisualizer:
    def __init__(self, root: Path, pickle_path: Path):
        """

        Args:
            root: Base folder for all Fisheye3DOD data.
        """
        self.root = root
        self.data_infos = mmengine.load(pickle_path)['data_list']
        self.tokens = [info['token'] for info in self.data_infos]
        self.tokens = self._filter_files(self.tokens, N=10)
        self.data_infos_by_dict = {info['token']: info for info in self.data_infos}
    
    def _filter_files(self, tokens, N):
        '''
            Only collect the first N images in each scene
        '''
        new_tokens = []
        scene_frame = defaultdict(list)
        for file_name in tokens:
            scene, ego, frame = file_name.split('.')
            scene_frame[scene].append(frame)
        for scene, frames in scene_frame.items():
            for frame in frames[:N]:
                new_tokens.append(f'{scene}.ego0.{frame}')
        return new_tokens
    
    def get_filepath(self, token: str, table: str) -> str:
        """For a token and table, get the filepath to the associated data.

        Args:
            token: Fisheye3DODVisualizer unique id.
            table: Type of table, for example image or velodyne.
            root: Base folder for all Fisheye3DOD data.

        Returns: Full get_filepath to desired data.

        """
        scene, ego, frame = token.split(".")
        filepath = f"{scene}/{ego}/{table}/{frame}.bin"
        filepath = str(self.root / filepath)
        return filepath

    def get_pointcloud(self, token: str) -> LidarPointCloud:
        """Load up the point cloud for a sample.

        Args:
            token: Fisheye3DODVisualizer unique id.
        Returns: LidarPointCloud for the sample in the Fisheye3DOD Lidar frame.

        """
        pc_filename = self.get_filepath(token, table='lidar')

        # The lidar PC is stored in the Fisheye3DOD LIDAR coord system.
        pc = LidarPointCloud(np.fromfile(pc_filename, dtype=np.float32).reshape(-1, 4).T)

        return pc

    def get_boxes(
            self, 
            token: str, 
            filter_classes: List[str] = None, 
            max_dist: float = None, 
            annos_type: str = 'gt',
            score_th: float = None) -> List[Box]:
        """Load up all the boxes associated with a sample.
            Boxes are in nuScenes lidar frame.

        Args:
            token: Fisheye3DODVisualizer unique id.
            filter_classes: List of Fisheye3DOD classes to use or None to use all.
            max_dist: List of Fisheye3DOD classes to use or None to use all.

        Returns: Boxes in nuScenes lidar reference frame.

        """

        annos = self.data_infos_by_dict[token]['annos']
        if annos_type == 'gt':
            instances_names = annos['gt_names']
            instances_boxes = annos['gt_boxes']
            scores = [np.nan] * len(instances_names)

        elif annos_type == 'pred':
            instances_names = annos['pred_names']
            instances_boxes = annos['pred_boxes']
            scores = annos['pred_scores']

        # filter classes
        if filter_classes is not None:
            classes_mask = np.array([n in filter_classes for n in instances_names], dtype=bool)
            instances_names = instances_names[classes_mask]
            instances_boxes = instances_boxes[classes_mask]
        
        # filter by max_dist
        if max_dist is not None:
            dists = np.linalg.norm(instances_boxes[:, :2], axis=1)
            dists_mask = dists <= max_dist
            instances_names = instances_names[dists_mask]
            instances_boxes = instances_boxes[dists_mask]
        
        # filter by score
        if score_th is not None and annos_type == 'pred':
            score_mask = scores >= score_th
            scores = scores[score_mask]
            instances_names = instances_names[score_mask]
            instances_boxes = instances_boxes[score_mask]
        
        center = instances_boxes[:, :3]
        lwh = instances_boxes[:, 3:6]  # length, width, height
        wlh = lwh[:, [1, 0, 2]]  # width, length, width
        rot_z = instances_boxes[:, 6]
        boxes = []
        for i in range(instances_boxes.shape[0]):
            quat_box = Quaternion(axis=(0, 0, 1), angle=rot_z[i])
            box = Box(center[i], wlh[i], quat_box, name=instances_names[i])
            
            # Set score or NaN.
            box.score = scores[i]
            # Set dummy velocity.
            box.velocity = np.array((0.0, 0.0, 0.0))
            boxes.append(box)

        return boxes

    def render_sample_data(
        self,
        token: str,
        sensor_modality: str = "lidar",
        with_anns: bool = True,
        axes_limit: float = 30,
        ax: Axes = None,
        view_3d: np.ndarray = np.eye(4),
        color_func: Any = None,
        augment_previous: bool = False,
        box_linewidth: int = 2,
        filter_classes: List[str] = None,
        max_dist: float = None,
        annos_type: str = 'gt',
        out_path: str = None,
    ) -> None:
        """Render sample data onto axis. Visualizes lidar in nuScenes lidar frame and camera in camera frame.

        Args:
            token: Fisheye3DOD token.
            sensor_modality: The modality to visualize, e.g. lidar or camera.
            with_anns: Whether to draw annotations.
            axes_limit: Axes limit for lidar data (measured in meters).
            ax: Axes onto which to render.
            view_3d: 4x4 view matrix for 3d views.
            color_func: Optional function that defines the render color given the class name.
            augment_previous: Whether to augment an existing plot (does not redraw pointcloud/image).
            box_linewidth: Width of the box lines.
            filter_classes: Optionally filter the classes to render.
            max_dist: Maximum distance in meters to still draw a box.
            out_path: Optional path to save the rendered figure to disk.
            render_2d: Whether to render 2d boxes (only works for camera data).

        """
        assert sensor_modality in ["lidar", "camera"], "Invalid sensor modality, sensor_modality must be 'lidar' or 'camera'."
        assert annos_type in ['gt', 'pred'], "Invalid annos type, annos_type must be 'gt' or 'pred'."

        # Default settings.
        if color_func is None:
            color_func = get_color

        boxes = self.get_boxes(
                               token, 
                               filter_classes=filter_classes, 
                               max_dist=max_dist, 
                               annos_type=annos_type, 
                               score_th=0.1
                            )  # In nuScenes lidar frame.

        if sensor_modality == "lidar":
            # Load pointcloud.
            pc = self.get_pointcloud(token)  # In Fisheye3DOD lidar frame.
            intensity = pc.points[3, :]

            # Project points to view.
            points = view_points(pc.points[:3, :], view_3d, normalize=False)

            # points, boxes = rotation_points_and_boxes_by_z_axis(points.T, boxes, -np.pi/4)

            coloring = intensity

            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            if not augment_previous:
                ax.scatter(points[0, :], points[1, :], c=coloring, s=1)
                ax.set_xlim(-axes_limit, axes_limit)
                ax.set_ylim(-axes_limit, axes_limit)

            if with_anns:
                for box in boxes:
                    color = np.array(color_func(box.name)) / 255
                    box.render(ax, view=view_3d, colors=(color, color, color), linewidth=box_linewidth)
        else:
            raise ValueError("Unrecognized modality {}.".format(sensor_modality))

        # ax.axis("off")
        # ax.set_title(token)
        ax.set_aspect("equal")

        # Render to disk.
        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path)
    
    def render_all(self,
                   sensor_modality: str = "lidar",
                   with_anns: bool = True,
                   axes_limit: float = 30,
                   ax: Axes = None,
                   view_3d: np.ndarray = np.eye(4),
                   color_func: Any = None,
                   augment_previous: bool = False,
                   box_linewidth: int = 2,
                   filter_classes: List[str] = None,
                   max_dist: float = None,
                   annos_type: str = 'gt',
                   output_dir: str = None,
                   ) -> None:
        """Render all samples in the dataset.
        """
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        for token in tqdm(self.tokens):
            out_path = None if output_dir is None else Path(output_dir) / f"{token}.png"
            self.render_sample_data(
                token=token,
                sensor_modality=sensor_modality,
                with_anns=with_anns,
                axes_limit=axes_limit,
                ax=ax,
                view_3d=view_3d,
                color_func=color_func,
                augment_previous=augment_previous,
                box_linewidth=box_linewidth,
                filter_classes=filter_classes,
                max_dist=max_dist,
                annos_type=annos_type,
                out_path=out_path
            )
            if output_dir is None:
                plt.show()
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fisheye3DODVisualizer")
    parser.add_argument("--root", type=str, required=True, help="Base folder for all Fisheye3DOD data.")
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to the pickle file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the rendered images.")
    parser.add_argument("--token", type=str, help="Specific token to render.")
    parser.add_argument("--sensor_modality", type=str, default="lidar", choices=["lidar", "camera"], help="Sensor modality to visualize.")
    parser.add_argument("--with_anns", type=bool, default=True, help="Whether to draw annotations.")
    parser.add_argument("--axes_limit", type=float, default=48, help="Axes limit for lidar data (measured in meters).")
    parser.add_argument("--annos_type", type=str, default='pred', choices=['gt', 'pred'], help="Annotations type.")
    parser.add_argument("--out_path", type=str, help="Path to save the rendered figure to disk.")
    
    args = parser.parse_args()

    root = Path(args.root)
    pickle_path = Path(args.pickle_path)
    vis = Fisheye3DODVisualizer(root, pickle_path)
    
    if args.token:
        vis.render_sample_data(
            token=args.token,
            sensor_modality=args.sensor_modality,
            with_anns=args.with_anns,
            axes_limit=args.axes_limit,
            annos_type=args.annos_type,
            out_path=args.out_path
        )
        plt.show()
    else:
        vis.render_all(
            sensor_modality=args.sensor_modality,
            with_anns=args.with_anns,
            axes_limit=args.axes_limit,
            annos_type=args.annos_type,
            output_dir=args.output_dir
        )
        
    