import os
import glob
import json
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from tqdm import tqdm  # Progress bar library
import time
from scipy.spatial import cKDTree  # Import cKDTree from scipy
from sklearn.cluster import DBSCAN

start_time = time.time()
# 경고 메시지 끄기
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def process_pcd_files_with_boxes(pcd_file_path_1, pcd_file_path_2, output_file_path):
    # PCD 파일 로드 및 다운샘플링
    pcd1 = o3d.io.read_point_cloud(pcd_file_path_1)
    pcd2 = o3d.io.read_point_cloud(pcd_file_path_2)

    voxel_size = 0.2
    pcd1_downsampled = pcd1.voxel_down_sample(voxel_size)
    pcd2_downsampled = pcd2.voxel_down_sample(voxel_size)

    # NumPy 배열 변환
    points1 = np.asarray(pcd1_downsampled.points)
    points2 = np.asarray(pcd2_downsampled.points)

    # cKDTree를 사용하여 포인트 필터링 (겹치는 점 제거)
    tree = cKDTree(points2)
    threshold = 0.2
    mask_to_remove = np.array([False] * len(points1))

    for i in range(len(points1)):
        dist, _ = tree.query(points1[i], k=1)
        if dist <= threshold:
            mask_to_remove[i] = True

    # 겹치는 점을 제외한 포인트
    filtered_points = points1[~mask_to_remove]

    # 필터링된 포인트를 포인트 클라우드로 변환
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 노이즈 제거 (통계적 방법)
    _, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1)
    denoised_pcd = filtered_pcd.select_by_index(ind)

    # 다시 NumPy 배열로 변환
    denoised_points = np.asarray(denoised_pcd.points)

    # 클러스터링 진행 (DBSCAN)
    eps_3d = 0.3
    min_samples_3d = 7
    clustering_3d = DBSCAN(eps=eps_3d, min_samples=min_samples_3d).fit(denoised_points)
    labels_3d = clustering_3d.labels_

    # 박스 정보 생성
    boxes_info = []
    unique_labels_3d = set(labels_3d)
    colors = np.random.rand(len(unique_labels_3d), 3)

    for label_3d in unique_labels_3d:
        if label_3d == -1:
            continue  # 노이즈 클러스터 제외

        cluster_indices_3d = (labels_3d == label_3d)
        cluster_points_3d = denoised_points[cluster_indices_3d]

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points_3d)
        bbox_3d = cluster_pcd.get_axis_aligned_bounding_box()

        bbox_height = bbox_3d.get_extent()[2]

        if bbox_height >= 0.08:
            bbox_points = bbox_3d.get_box_points()
            bbox_data = {
                "label": int(label_3d),
                "height": float(bbox_height),
                "points": [[float(p[0]), float(p[1]), float(p[2])] for p in bbox_points],
                "color": list(colors[label_3d][:3])
            }
            boxes_info.append(bbox_data)

    # 결과 저장
    with open(output_file_path, 'w') as json_file:
        json.dump(boxes_info, json_file, indent=4)
    
# Example usage
# process_pcd_files_with_boxes("path_to_pcd1.pcd", "path_to_pcd2.pcd", "output_boxes.json")

def process_all_pcd_files_in_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a sorted list of all PCD files in the input folder
    pcd_files = sorted(glob.glob(os.path.join(input_folder, "*.pcd")))
    
    # Process each PCD file pair with a progress bar
    for i in tqdm(range(len(pcd_files) - 1), desc="Processing PCD files"):  # Use tqdm for progress tracking
        pcd_file_1 = pcd_files[i]
        if i > 100 :
            pcd_file_2 = pcd_files[1]
        else :
            pcd_file_2 = pcd_files[i + 100]
        
        # Generate output JSON file name based on the first PCD file name
        base_name = os.path.basename(pcd_file_1).replace(".pcd", ".json")
        output_file_name = os.path.join(output_folder, base_name)
        
        # Call the function to process the files
        process_pcd_files_with_boxes(pcd_file_1, pcd_file_2, output_file_name)


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
    input_folder = processfilelist[w]
    output_folder = outputfilelist[w]
    print("진행중 " + str(w+1) + "/7")
    process_all_pcd_files_in_folder(input_folder, output_folder)

print("오류율 측정중")

def analyze_json_files(directory):
    total_json_count = 0
    non_empty_json_count = 0

    # 전체 파일 개수 미리 계산
    all_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('.json')]
    total_json_count = len(all_files)

    if total_json_count == 0:
        print("JSON 파일이 없습니다.")
        return total_json_count, non_empty_json_count

    # tqdm으로 진행률 표시
    for file_path in tqdm(all_files, desc="JSON 파일 처리 중", unit="파일"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

                # 내용이 비어있지 않고, [] (빈 리스트)가 아닌 경우
                if content and content != []:
                    non_empty_json_count += 1
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"파일을 처리할 수 없습니다: {file_path}, 에러: {e}")

    return total_json_count, non_empty_json_count

# 사용 예시
directory_path = "/Users/iyeongchan/Desktop/Projrct/COSE416_HW1_tutorial/data/"  # A폴더 경로
total_count, non_empty_count = analyze_json_files(directory_path)

if total_count > 0:
    percentage = (non_empty_count / total_count) * 100
    print(f"\n총 JSON 파일 개수: {total_count}")
    print(f"값이 비어있지 않은 JSON 파일 개수: {non_empty_count}")
    print(f"비율: {percentage:.2f}%")
else:
    print("JSON 파일이 없습니다.")

