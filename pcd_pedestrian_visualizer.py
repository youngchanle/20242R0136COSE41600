import open3d as o3d
import json
import numpy as np

def visualize_single_pcd_with_json(pcd_file, json_file):
    """Load and visualize a single PCD file along with its bounding boxes from a JSON file."""
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Apply gradient coloring from gray to black based on Z values
    points = np.asarray(pcd.points)
    z_values = points[:, 2]
    min_z, max_z = z_values.min(), z_values.max()
    colors = 0.5 * (1 - (z_values - min_z) / (max_z - min_z))  # Grayscale gradient
    pcd.colors = o3d.utility.Vector3dVector(np.tile(colors[:, None], (1, 3)))

    # Load bounding boxes from JSON
    bounding_boxes = []
    with open(json_file, 'r') as f:
        bbox_data = json.load(f)
        for bbox_info in bbox_data:
            min_bound = np.min(bbox_info['points'], axis=0)
            max_bound = np.max(bbox_info['points'], axis=0)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            bbox.color = [0, 1, 0]  # Green color
            bounding_boxes.append(bbox)

    # Visualize
    o3d.visualization.draw_geometries([pcd, *bounding_boxes],
                                      window_name="PCD and Bounding Boxes",
    )

# Specify file paths
pcd_path = "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/03_straight_crawl/pcd/pcd_000888.pcd"
json_path = "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/03_straight_crawl/json/pcd_000888.json"

# Visualize
visualize_single_pcd_with_json(pcd_path, json_path)