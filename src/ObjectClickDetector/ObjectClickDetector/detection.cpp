#include "detection.h"
#include <opencv2/opencv.hpp>

void parse_yolo_output(const torch::Tensor& output)
{
    detected_boxes.clear();

    
    detected_boxes.push_back({
        cv::Rect(100, 100, 150, 150),
        0,
        0.9f
        });
}

void draw_boxes(cv::Mat& img)
{
    for (auto& d : detected_boxes) {
        cv::rectangle(img, d.box, cv::Scalar(0, 255, 0), 2);
    }
}
