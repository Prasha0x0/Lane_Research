# Adaptive Lightweight Lane Detection for Autonomous Vehicles Under Varying Illumination Conditions

A lightweight, training-free computer vision approach for lane detection under normal, shadowed, and low-light conditions, designed with computational efficiency and resource-constrained autonomous systems in mind.

## Overview

Lane detection is a fundamental component of autonomous driving systems, enabling vehicles to understand road boundaries and maintain their position within a driving lane. However, variations in illumination, shadows, and low-light environments can significantly affect the reliability of conventional lane detection techniques.

This research investigates an adaptive, lightweight lane detection framework that combines HSV-based color thresholding, spatially weighted lane estimation, and illumination-specific image enhancement techniques.

Unlike deep learning-based approaches that require extensive training data and computational resources, the proposed method follows a classical computer vision approach without requiring model training or GPU acceleration.

The primary objective is to develop an efficient and adaptable lane detection method suitable for real-time applications on computationally constrained hardware.

## Research Objectives

- Develop a lightweight, training-free lane detection framework.
- Improve lane detection reliability under varying illumination conditions.
- Investigate the effects of shadows and low-light environments on lane detection.
- Incorporate spatial information to improve lane position estimation.
- Evaluate detection performance against conventional lane detection approaches.
- Explore computational efficiency for real-time autonomous driving applications.

## Proposed Methodology

The proposed framework consists of multiple processing stages designed to adapt to different illumination conditions.

### 1. Adaptive HSV-Based Lane Detection

HSV color space is used to identify lane markings based on their color and brightness characteristics. Adaptive thresholding helps distinguish lane markings from surrounding road regions under varying illumination conditions.

### 2. Two-Zone Weighted Centroid Estimation

The image is divided into two horizontal regions to estimate lane position using spatial information.

- **Lower zone (55–100%):** Assigned a weight of 70% to prioritize nearby lane markings.
- **Upper zone (25–55%):** Assigned a weight of 30% to incorporate additional directional information.

The weighted centroid approach helps estimate the lane direction while reducing the influence of distant or irrelevant image regions.

### 3. Low-Light Enhancement

For challenging low-light conditions, an additional image enhancement pipeline is investigated using:

- Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB color space.
- Morphological Top-Hat transformation.
- Geometric filtering based on contour properties.
- Principal Component Analysis (PCA) for orientation-based filtering.

These operations help enhance potential lane markings while suppressing irrelevant image regions.

### 4. Lane Direction Classification

Based on the estimated lane position and direction, the system produces the following outputs:

- `LEFT`
- `RIGHT`
- `FORWARD`
- `LANE LOST`
- `NO LANE`

## Experimental Evaluation

The proposed approach is evaluated against conventional lane detection techniques, including Hough-based detection and fixed HSV thresholding.

### Comparative Performance on CULane

| Method | Accuracy (%) | F1 Score | IoU |
|---|---:|---:|---:|
| Hough Transform | 68.8 | 10.0 | 8.6 |
| Fixed HSV Thresholding | 55.4 | 9.7 | 8.0 |
| Proposed Method | **71.7** | **10.0** | **8.8** |

### Nighttime Performance

| Method | Accuracy (%) |
|---|---:|
| Hough Transform | 54.1 |
| Fixed HSV Thresholding | 36.9 |
| Proposed Method | **70.8** |

The experimental results indicate that the proposed approach achieves improved detection performance compared with the evaluated conventional methods, particularly under nighttime conditions.

### Computational Efficiency

The proposed framework is designed to operate without GPU acceleration or model training.

| Processing Method | Execution Time | Throughput |
|---|---:|---:|
| Method 2 | 0.9 ms | 1156 FPS |
| Method 3 | 1.5 ms | 686.8 FPS |

These measurements demonstrate the computational efficiency of the proposed image processing pipeline under the evaluated experimental setup.

## Key Contributions

- A lightweight, training-free lane detection framework based on classical computer vision techniques.
- Adaptive HSV-based lane extraction for varying illumination conditions.
- A two-zone weighted centroid approach for spatially informed lane direction estimation.
- A specialized low-light enhancement pipeline combining CLAHE, morphological operations, and geometric filtering.
- Comparative evaluation against conventional lane detection methods.
- Investigation of computational efficiency for resource-constrained autonomous driving applications.

## Technologies and Techniques

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Image Processing:** HSV Thresholding, CLAHE, Morphological Operations
- **Geometric Analysis:** Contour Filtering, Principal Component Analysis
- **Evaluation:** CULane Dataset, Accuracy, F1 Score, Intersection over Union (IoU)

## Research Paper

The complete research paper, including the methodology, experimental setup, comparative evaluation, and results, is available below.

**[Read the Research Paper (PDF)](lanedetection.pdf)**

## Future Work

Potential directions for further investigation include:

- Evaluation across more diverse road environments and illumination conditions.
- Further validation on resource-constrained embedded platforms.
- Improving robustness against complex road markings, occlusions, and varying weather conditions.
- Exploring integration with other perception modules for autonomous navigation.
