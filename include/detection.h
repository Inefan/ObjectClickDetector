#pragma once
#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>   
#include "detect_types.h"

extern std::vector<DetectBox> boxes;

void parse_yolo_output(const torch::Tensor& output, int orig_h, int orig_w);
void draw_boxes(cv::Mat& img);