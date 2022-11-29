# Target re-identification based on topological 2D and 3D features

Matching and re-identification of multiple targets across two cameras using topological features.

This is a work in progress and is not experimentally verified yet.

The algorithm extends on Xudong et al (2022)'s triangular topological sequence to match similar targets based on their relative positions and angles. Estimation of 3D coordinates based on target size is introduced from Seah et al (2022).

# Dependencies

* pytorch
* matplotlib
* numpy
* scipy
* OpenCV

# Usage

Run "detectobjects.py" on each image to detect targets with Yolo V5. Make sure the path to the image is specified in the script. See example output images in "data/samples" folder. Detected objects and their coordinates will be output on the terminal.

Copy the terminal output into a csv file and manually clean the data. Store the csv file in the "csv" folder. See example csv files in "data/samples" folder.

Run any of the "reid" scripts to generate topography map and re-identify targets across cameras. See each script for a description of how the algorithm works.

# References

Li, X.; Wu, L.; Niu, Y.; Ma, A. Multi-Target Association for UAVs Based on Triangular Topological Sequence. Drones 2022, 6, 119. [Link](https://doi.org/10.3390/drones6050119)

Seah, S.X.; Lau, Y.H.; Srigrarom, S. Multiple Aerial Targets Re-Identification by 2D- and 3D- Kinematics-Based Matching. J. Imaging 2022, 8, 26. [Link](https://doi.org/10.3390/jimaging8020026)

SPOT-IT-3D Github code [Link](https://github.com/seahhorse/spot-it-3d)