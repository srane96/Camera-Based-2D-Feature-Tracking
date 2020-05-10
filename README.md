# Camera-Based 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## MP.1 Data Buffer Optimization
The goal of this section was to implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This was achieved by removing the first eleent everytime size of data buffer exceeded the max limit. (line 68-69)

## MP.2 Keypoint Detection
The goal of this section was to implement different detectors (HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT) and make them selectable by setting a string accordingly. For this, different string names are created (line 81-87), and depending upon the uncommented string, a different function is called (line 93-104). In case of Harris-Corner detection, custom non-max supression code was implemented.(matching2D_student.cpp line 163 - 197)

## MP.3 Keypoint Removal
The goal of this section was to remove all keypoints outside of the car which is in front of the camera. For this a cv::Rect was defined roughly at the center of the image, and only those keypoints were kept which were inside this rectangle (line 114-124)

## MP.4 Keypoint Descriptors
The goal of this section was to implement different types of descriptors (BRIEF, ORB, FREAK, AKAZE and SIFT) and make them selectable by setting a string accordingly. For this, different string names are created (line 81-87), and depending upon the uncommented string, a different function is called (line 93-104)

## MP.5 Descriptor Matching
In this section FLANN matching as well as k-nearest neighbor selection were implemented. 

## MP.6 Descriptor Distance Ratio	
In this section K-Nearest-Neighbor matching with K=2 was implemented along with the descriptor distance ratio test, which looks at the ratio of best vs. second-best match and keeps the match if only their ratio is less than 0.8.

## MP.7 Performance Evaluation 1
In this section all the different keypoint detectors were run on all the image sequences and following attributes were compared. <b>Keypoint Detector Name</b>, <b>Total Keypoints Detected</b>, <b> Keypoints on the Vehicle</b>, <b> Neighbourhood size</b>.
Output Spreadsheet - [MP7 Performance Eval](outputs/MP7. KeyPoint Detectors - output.csv)

## MP.8 MP.9 Performance Evaluation 2 & 3
In this section all the possible combinations of the detectors and descriptors are compared. And for each combination, detector name, detector time, descriptor name and descriptor time was logged.
Output Spreadsheet - [MP8 and MP9 Performance Eval](outputs/MP.8 and MP.9 Detectors and Descriptors - output2.csv)

