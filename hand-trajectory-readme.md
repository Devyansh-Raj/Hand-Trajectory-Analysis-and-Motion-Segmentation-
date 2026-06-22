# Hand Trajectory Analysis & Motion Segmentation

A computer vision pipeline that tracks hand movement frame-by-frame and automatically segments pick-and-place actions into distinct motion phases — built as groundwork for action-level understanding in imitation learning.

## Problem

Before you can teach a model to imitate a human demonstration, you need to know *where the action boundaries are* — when does "reach" end and "grasp" begin? Raw video gives you pixels, not phases. This project builds the bridge: from raw hand-tracking data to labeled motion segments.

## Approach

1. **Landmark extraction:** MediaPipe extracts 21 hand landmarks per frame; OpenCV handles video I/O and visualization.
2. **Trajectory smoothing:** cubic spline interpolation over 500 resampled points per trajectory, removing frame-rate artifacts.
3. **Velocity estimation + Gaussian smoothing:** to get a clean signal of how fast the hand/wrist is moving at each point in time.
4. **Segmentation:** prominence-aware valley detection on the velocity signal — motion naturally slows at phase transitions (e.g., right before a grasp), and these valleys mark segment boundaries.
5. **Baseline classifier:** a primitive-labeling classifier achieving 71% accuracy on segmenting pick-and-place phases.

## Stack

MediaPipe · OpenCV · SciPy (spline interpolation, signal processing) · NumPy

## Results

71% baseline accuracy on primitive motion-phase labeling, using unsupervised signal-processing techniques (no training data required for segmentation itself).

## What I'd improve next

- Compare against a learned segmentation model (e.g., a small TCN) to see how much the unsupervised baseline leaves on the table.
- Extend from single-hand to bimanual trajectories.

## Run it

```bash
git clone https://github.com/Devyansh-Raj/Hand-Trajectory-Analysis-and-Motion-Segmentation-
cd Hand-Trajectory-Analysis-and-Motion-Segmentation-
pip install -r requirements.txt
python main.py --video path/to/video.mp4
```

(Update the run command to match your actual entry point.)
