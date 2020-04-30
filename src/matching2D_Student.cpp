#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_HOG") ? cv::NORM_L2 : cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descriptorType.compare("DES_HOG") == 0)
        {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);   
        }
        else
        { 
            matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        }
        cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if((*it).size() == 2)
            {
                if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
                {
                    matches.push_back((*it)[0]);
                }
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        cout << "Using BRIEF descriptor" << endl;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        cout << "Using ORB descriptor" << endl;
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        cout << "Using FREAK descriptor" << endl;
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        cout << "Using AKAZE descriptor" << endl;
        extractor = cv::AKAZE::create();
    }
    else
    {
        cout << "Using SIFT descriptor" << endl;
        extractor = cv::xfeatures2d::SIFT::create();
    }
    
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    std::cout << "neighbourhood size " << keypoints[0].size << std::endl;
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in the image using Harris corner detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cout << "Using HARRIS detector" << endl;
    double t = (double)cv::getTickCount();
    int blockSize = 4; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1 );
    cv::cornerHarris( img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT ); 
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    
    double maxOverlap = 0.0; 
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { 
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     
                    keypoints.push_back(newKeyPoint); 
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    std::cout << "neighbourhood size " << keypoints[0].size << std::endl;
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string name = "Harris Corner";
        cv::namedWindow(name);
        cv::imshow(name, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    double t = (double)cv::getTickCount();
    if (detectorType.compare("FAST") == 0)
    {
        // Fast
        cout << "Using FAST detector" << endl;
        int threshold = 30;
        bool NMS = true;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, NMS, cv::FastFeatureDetector::TYPE_9_16);
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        // Orb
        cout << "Using ORB detector" << endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 8, 30);
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        // Akaze
        cout << "Using AKAZE detector" << endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        // Brisk
        cout << "Using Brisk detector" << endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        detector->detect(img, keypoints);
    }
    else
    {
        // Sift
        cout << "SIFT detector" << endl;
        cv::Ptr<cv::FeatureDetector> sift_detector = cv::xfeatures2d::SIFT::create();
        sift_detector->detect(img, keypoints);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << "ms" << std::endl;
    std::cout << "neighbourhood size " << keypoints[0].size << std::endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(detectorType);
        cv::imshow(detectorType, visImage);
        cv::waitKey(0);
    }
}