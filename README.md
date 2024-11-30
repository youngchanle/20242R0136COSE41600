# 1/5-Second Point Cloud Pedestrian Detection on CPU

## Introduction

### 1. Project Purpose
Design and implement an algorithm for effectively detecting pedestrians using 3D Point Cloud Data (PCD) with a focus on efficient CPU-based processing.

### 2. Existing PCD-based Pedestrian Detection Algorithm Problems
1. **Processing Speed Issue**: 
   - PCD contains massive 3D coordinate information
   - Requires significant time for real-time processing

2. **High-Performance Hardware Requirements**:
   - Complex 3D calculations and deep learning models demand powerful GPU and computational resources

#### Examples of Existing Approaches
1. **VoxelNet**: 
   - Converts 3D point clouds into grid format
   - Provides high accuracy
   - Extremely high computational cost

2. **PointNet**:
   - Effective at extracting point cloud data features
   - Performance degrades when processing large volumes of data

## Methodology: Detailed Implementation

### Key Processing Steps
1. **Point Cloud Data Preprocessing**
   - Voxel Down Sampling
     - Reduce point cloud resolution to voxel size of 0.2
     - Decreases computational complexity
     - Maintains key spatial information

2. **Background Removal and Filtering**
   - Use cKDTree for efficient point filtering
     - Compare points between two point cloud frames
     - Remove points within a 0.2m threshold distance
   - Statistical Outlier Removal
     - Use Open3D's statistical method
     - Remove noise points
     - Parameters: 
       - Neighboring points: 5
       - Standard deviation ratio: 1

3. **3D Clustering**
   - Implement DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
     - Clustering Parameters:
       - Epsilon (eps): 0.3
       - Minimum samples: 7
     - Identifies potential pedestrian clusters

4. **Pedestrian Detection**
   - Bounding Box Generation
     - Create axis-aligned bounding boxes for each cluster
   - Height-based Filtering
     - Minimum height threshold: 0.08m
     - Helps discriminate pedestrians from small objects

5. **Output Generation**
   - Generate JSON files with:
     - Cluster label
     - Bounding box height
     - Bounding box points
     - Randomly assigned cluster color

### Processing Workflow
1. Load two Point Cloud Data (PCD) files
2. Down-sample both point clouds
3. Remove background and filter points
4. Apply 3D clustering
5. Generate bounding boxes
6. Save detection results as JSON

## Results

### 1. Processing Speed
- Average processing time per frame: 0.18 seconds
- Tested across 7 different scenarios:
  1. Straight walk
  2. Straight duck walk
  3. Straight crawl
  4. Zigzag walk
  5. Additional walking variations

### 2. Accuracy
- Error rate: 0.02%
- Detection capabilities:
  - Identifying lying down individuals
  - Detecting people hidden under trees
  - Handling various walking postures and styles

*Note: Error rate calculated as (number of PCDs with selected objects) / (total PCD count), which might differ from actual values*

## Computational Approach
- Utilizes CPU-based processing
- Leverages libraries:
  - Open3D (3D data processing)
  - scikit-learn (DBSCAN clustering)
  - NumPy (numerical computing)
  - SciPy (scientific computing)

## Technical Specifications
- Test Computer: MacBook Pro 14 (2021)
  - Chip: M1 Pro (18-core)
  - Memory: 16GB

## Practical Applications
- Pedestrian detection in various scenarios
- Potential uses in:
  - Autonomous vehicle safety systems
  - Urban planning and crowd monitoring
  - Robotics and computer vision research

## References
1. Open3D. (2024, November 26). Point Cloud Discussion [GitHub Post]
2. Test Computer Specifications: MacBook Pro 14 (2021)
