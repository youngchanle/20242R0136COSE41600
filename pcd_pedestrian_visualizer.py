import open3d as o3d
import json
import numpy as np
import os
import re
import time

def extract_number(filename):
    """파일 이름에서 숫자를 추출"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def load_files(pcd_folder, json_folder):
    """폴더 내 파일을 정렬하여 반환"""
    pcd_files = sorted([f for f in os.listdir(pcd_folder) if f.endswith('.pcd')], key=extract_number)
    json_files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')], key=extract_number)
    return [(os.path.join(pcd_folder, p), os.path.join(json_folder, j))
            for p, j in zip(pcd_files, json_files) if extract_number(p) == extract_number(j)]

def load_geometry(pcd_file, json_file):
    """PCD 및 JSON 데이터를 로드하여 포인트 클라우드와 바운딩 박스 반환"""
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 회색에서 검정색으로 그라데이션 적용
    points = np.asarray(pcd.points)
    z_values = points[:, 2]  # Z축 값 사용
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    
    # Z값에 따른 색상 계산: 깊이가 클수록 검정색으로, 작을수록 회색으로
    colors = np.zeros_like(points)
    for i, z in enumerate(z_values):
        normalized_depth = (z - min_z) / (max_z - min_z)  # 깊이를 0~1로 정규화
        colors[i] = [0.5 * (1 - normalized_depth), 0.5 * (1 - normalized_depth), 0.5 * (1 - normalized_depth)]  # 회색에서 검정색으로 그라데이션

    pcd.colors = o3d.utility.Vector3dVector(colors)

    bounding_boxes = []

    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            bbox_data = json.load(f)
            for bbox_info in bbox_data:
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=np.min(bbox_info['points'], axis=0),
                    max_bound=np.max(bbox_info['points'], axis=0)
                )
                bbox.color = bbox_info['color']
                bounding_boxes.append(bbox)
    return pcd, bounding_boxes

def visualize_real_time_auto(pcd_folder, json_folder, update_interval=1.0):
    """PCD 및 JSON을 실시간으로 자동 업데이트하여 시각화"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)  # 창 크기 2배로 확대

    # 초기 파일 로드
    matched_files = load_files(pcd_folder, json_folder)
    current_index = 0
    pcd, bounding_boxes = load_geometry(*matched_files[current_index])

    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)

    def update_geometry():
        """다음 프레임 데이터를 로드하고 시각화"""
        nonlocal current_index, pcd, bounding_boxes
        current_index = (current_index + 1) % len(matched_files)

        # 다음 데이터 로드
        new_pcd, new_bboxes = load_geometry(*matched_files[current_index])

        # 기존 지오메트리를 제거하고 새로 추가
        vis.clear_geometries()
        vis.add_geometry(new_pcd)
        for bbox in new_bboxes:
            vis.add_geometry(bbox)
        view_ctl = vis.get_view_control()
        view_ctl.set_zoom(0.15)  # 줌 아웃으로 시야 확대
        view_ctl.set_lookat([-10, 50, 0])  # 중심점 설정
        view_ctl.set_front([0, -2, 1])  # 카메라 방향 설정
        view_ctl.set_up([0, 11, 0])  # 카메라 상단 방향 설정
        # 뷰포트 업데이트
        vis.poll_events()
        vis.update_renderer()

    # 자동 업데이트 루프
    print("\n자동으로 데이터가 업데이트됩니다. 창을 닫으면 종료됩니다.")
    try:
        while vis.poll_events():
            # 업데이트 주기마다 프레임 갱신
            time.sleep(update_interval)
            update_geometry()
    except KeyboardInterrupt:
        print("종료 중...")

    vis.destroy_window()

# Example usage
input_folder = "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/05_straight_duck_walk/pcd"
output_folder = "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/05_straight_duck_walk/json"
pcd_folder = input_folder
json_folder = output_folder

# 실행
visualize_real_time_auto(pcd_folder, json_folder, update_interval=0.1)