#pragma once
#include <vector>
#include "mouse_handler.h"
#include <torch/torch.h>

void parse_yolo_output(const torch::Tensor& output);
void draw_boxes(cv::Mat& img);
