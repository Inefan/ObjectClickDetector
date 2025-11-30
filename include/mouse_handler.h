#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct DetectBox {
    cv::Rect box;
    int cls;
    float score;
};


extern cv::Mat original_image;
extern int selected_box;
extern std::vector<DetectBox> detected_boxes;


void onMouse(int event, int x, int y, int flags, void* userdata);

