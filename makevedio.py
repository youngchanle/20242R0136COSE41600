
import open3d as o3d
import json
import numpy as np
import os
import re
import time
import cv2  # Import OpenCV for video handling

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
                # AxisAlignedBoundingBox 대신 LineSet으로 박스 생성
                min_bound = np.min(bbox_info['points'], axis=0)
                max_bound = np.max(bbox_info['points'], axis=0)
                
                # 박스의 8개 꼭짓점 정의
                corners = np.array([
                    [min_bound[0], min_bound[1], min_bound[2]],
                    [max_bound[0], min_bound[1], min_bound[2]],
                    [max_bound[0], max_bound[1], min_bound[2]],
                    [min_bound[0], max_bound[1], min_bound[2]],
                    [min_bound[0], min_bound[1], max_bound[2]],
                    [max_bound[0], min_bound[1], max_bound[2]],
                    [max_bound[0], max_bound[1], max_bound[2]],
                    [min_bound[0], max_bound[1], max_bound[2]]
                ])
                
                # 박스를 구성하는 12개의 선 정의
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # 바닥 사각형
                    [4, 5], [5, 6], [6, 7], [7, 4],  # 상단 사각형
                    [0, 4], [1, 5], [2, 6], [3, 7]   # 상하 연결선
                ]
                
                # 색상 설정
                colors = np.tile(bbox_info['color'], (len(lines), 1))

                # LineSet 생성
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                # 선의 두께 설정 (Open3D에서는 직접적인 선 두께 설정이 불가능하지만, 대신 'LineSet'의 스타일을 조정할 수 있음)
                # 이 부분은 OpenCV나 다른 라이브러리를 활용해 해결할 수 있음
                line_set.paint_uniform_color([0.0, 1.0, 0.0])  # 색상 설정 (기본: 녹색)

                bounding_boxes.append(line_set)

    return pcd, bounding_boxes
def visualize_real_time_auto(pcd_folder, json_folder, output_video_path, update_interval=1.0):
    """PCD 및 JSON을 실시간으로 자동 업데이트하여 시각화하고, 영상을 저장"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)  # 창 크기 2배로 확대

    # 초기 파일 로드
    matched_files = load_files(pcd_folder, json_folder)
    current_index = 0
    pcd, bounding_boxes = load_geometry(*matched_files[current_index])

    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)

    # 비디오 저장을 위한 설정 (MP4, H.264 코덱)
    frame_width = 1920
    frame_height = 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'는 mp4 파일을 위한 코덱
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))  # 30 FPS로 설정

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
        
        # 카메라 설정
        view_ctl = vis.get_view_control()
        view_ctl.set_zoom(0.3)  # 줌 아웃으로 시야 확대
        view_ctl.set_lookat([-10, 50, 0])  # 중심점 설정
        view_ctl.set_front([0, -2, 1])  # 카메라 방향 설정
        view_ctl.set_up([0, 11, 0])  # 카메라 상단 방향 설정
        
        # 뷰포트 업데이트
        vis.poll_events()
        vis.update_renderer()

        # 화면을 이미지로 캡처하고 비디오 파일에 추가
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        image = np.uint8(np.asarray(image) * 255)  # Convert to uint8
        out.write(image)  # 프레임을 비디오 파일에 추가

    # 자동 업데이트 루프
    print("\n자동으로 데이터가 업데이트되며, 영상이 저장됩니다. 창을 닫으면 종료됩니다.")
    try:
        while vis.poll_events():
            # 한 바퀴가 다 돌아간 후 저장하도록 설정
            update_geometry()
            time.sleep(update_interval)

            # 한 바퀴가 완료되었으면 비디오를 저장
            if current_index == 0:
                print("한 바퀴가 완료되었습니다. 영상 저장 중...")
                break
    except KeyboardInterrupt:
        print("종료 중...")

    # 비디오 파일을 저장하고 종료
    out.release()  # 비디오 저장 종료
    vis.destroy_window()



processfilelist = [
    
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/01_straight_walk/pcd",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/02_straight_duck_walk/pcd",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/03_straight_crawl/pcd",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/05_straight_duck_walk/pcd",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/06_straight_crawl/pcd",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/04_zigzag_walk/pcd",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/07_straight_walk/pcd"
]

outputfilelist = [
    
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/01_straight_walk/json",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/02_straight_duck_walk/json",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/03_straight_crawl/json",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/05_straight_duck_walk/json",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/06_straight_crawl/json",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/04_zigzag_walk/json",
    "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/07_straight_walk/json"
]
for w in range(len(processfilelist)) :
    pcd_folder = processfilelist[w]
    json_folder = outputfilelist[w]
    output_video_path = f"/Users/iyeongchan/Desktop/Projrct/upload/{w+1}.mp4"
    print("진행중 " + str(w+1) + "/7")
    visualize_real_time_auto(pcd_folder, json_folder, output_video_path, update_interval=0)


