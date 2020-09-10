# COMP4905_SfM
3d reconstruction using image-based incremental structure from motion

# Abstract 
This project presents an alternative method for incremental structure from motion (SfM) with histogram comparison to improve computation efficiency and reconstruction accuracy. Currently global SfM and incremental/sequential SfM are two main methods in image-based 3d reconstruction. In general, incremental SfM are slower than global SfM because it implements random sample consensus (RANSAC) for outlier filtering and bundle adjustment for 3d point clouds optimization [14]. But incremental SfM has better robustness and often gets pretty good solutions, so it is significant to improve its efficiency but remaining high accuracy of result.

We can reduce its time complexity of image matching through optimizing the matching relationship using color histogram-based comparison. It compares the histograms of each image to get the similarity correlation and simplifies the matching relationship to reduce redundant calculation in feature matching and triangulation. The method draws upon a minimum spanning tree (MST) algorithm to construct a graph based on the adjacency matrix of similarity. Then, we simplify the matching relationships of images in which the connections with high similarity are reserved and other connections are dismissed. Only matching the images with high similarity can still get abundant good matches to reconstruct 3d structure in triangulation. In addition, this paper explains how to use functions within OpenCV (3.4.10) for incremental SfM systems and use MeshLab to visualize the results.

# Setup (for Mac OS)
1. (optional) install homebrew
2. run the following commands in terminal:
```
brew install opencv@3.4.10
brew install cmake
```

# Running the program
1. Open the terminal then run the following commands in the path you want to load the project
```
cd ./~/COMP4905_sfm
mkdir build
cd build
cmake ..
make
- ./structure_from_motion
```
2. To exit the program, press any key after the reconstruction was successful


# Results









