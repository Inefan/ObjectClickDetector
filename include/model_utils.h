#pragma once
#include <torch/script.h>
#include <opencv2/opencv.hpp>

extern bool MODEL_LOADED;

torch::jit::script::Module load_model(const std::string& path);
torch::Tensor preprocess_image(const cv::Mat& img_original);
torch::Tensor run_inference(torch::jit::script::Module& model, const cv::Mat& img);
