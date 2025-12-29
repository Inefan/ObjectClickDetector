#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "detect_types.h"

class Detector;

struct MouseContext {
	Detector* detector;
	cv::Mat* base;
};



void onMouse(int event, int x, int y, int flags, void* userdata);

