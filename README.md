# 1/5-Second Point Cloud Pedestrian Detection on CPU

### 1. Project Purpose
Design and implement an algorithm for effectively detecting pedestrians using 3D Point Cloud Data (PCD).

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

## Methodology: Solution Approach

### Key Steps
1. **Background Removal**:
   - Remove unnecessary points (fixed structures, vehicles)
   - Focus on regions of interest (pedestrians)
   - Removal method involves comparing with previous frames

### Detailed Process
1. **3D PCD Down Sampling**:
   - Reduce input PCD and first frame (or 100 frames later) to voxel size 0.2

2. **Background Removal**:
   - Overlay two PCDs
   - Remove close points, leaving potential object points
   - Uses cKDTree
   - Perform noise removal on remaining points

3. **3D Clustering**:
   - Conduct 3D clustering to detect pedestrians
   - Uses DBSCAN algorithm

## Results

### 1. Processing Speed
- Average processing time per frame: 0.18 seconds
- Tested across 7 different scenarios

### 2. Accuracy
- Error rate: 0.02%
- Capable of detecting:
  - Lying down individuals
  - People hidden under trees

*Note: Error rate calculated as (number of PCDs with selected objects) / (total PCD count), which might differ from actual values*

## Technical Specifications
- Test Computer: MacBook Pro 14 (2021)
  - Chip: M1 Pro (18-core)
  - Memory: 16GB

## References
1. Open3D. (2024, November 26). Point Cloud Discussion [GitHub Post]
2. Test Computer Specifications
