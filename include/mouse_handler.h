#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "detect_types.h"


extern cv::Mat original_image;
extern int selected_box;
extern std::vector<DetectBox> boxes;


void onMouse(int event, int x, int y, int flags, void* userdata);

