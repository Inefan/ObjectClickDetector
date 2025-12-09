#pragma once
#include <opencv2/opencv.hpp>

struct DetectBox {
    cv::Rect box;
    int cls;
    float score;
};
