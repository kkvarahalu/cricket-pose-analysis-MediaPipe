# AI/ML Computer Vision – Sports Biomechanics Assignment

## Objective
Build a pose-based movement analysis pipeline for a cricket player using a side-on phone video and extract interpretable biomechanics metrics.

## Video Selection
I used a side-on cricket batting video recorded using a phone camera.
This view is suitable because major joints (elbow, knee, hip) are clearly visible and depth ambiguity is minimized.
However, fast bat movement and partial occlusion can introduce minor keypoint noise.

## Model Used
**MediaPipe Pose**
- Provides 33 body keypoints
- Lightweight and fast on CPU
- Suitable for single-person sports motion analysis

## Pipeline Overview
1. Read video frame-by-frame using OpenCV
2. Apply MediaPipe Pose to extract body keypoints
3. Draw skeleton overlay on the video
4. Save frame-wise keypoints to CSV
5. Compute joint angles (elbow, knee, hip)
6. Calculate interpretable metrics (ROM and stability)

## Metrics Extracted
### 1. Elbow Angle
Represents arm flexion and extension during batting.
Important for shot control and swing mechanics.

### 2. Knee Angle
Represents front-leg bend and stability.
Important for balance and weight transfer.

### 3. Hip Angle
Represents torso–leg alignment.
Important for posture and power generation.

### Range of Motion (ROM)
Computed as max angle − min angle.
Shows how much a joint moves during the action.

### Stability (Variance)
Variance of angle values over time.
Lower variance indicates smoother and more stable motion.

## Observations & Limitations
- Minor jitter during fast motion
- Occlusion due to bat and net
- 2D pose limits depth accuracy

## Improvement Plan
- Apply temporal smoothing to keypoints
- Crop ROI to track only the player
- Compare MediaPipe with YOLO Pose or RTMPose
- Collect multi-angle and multi-player data

## Data Strategy (If Training a Model)
- Split data by player, not frames
- Train: 70%, Validation: 15%, Test: 15%

## Evaluation Strategy
- Keypoint stability
- Missed detection rate
- Smoothness of angle curves
- Visual quality of skeleton overlay
