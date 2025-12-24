#pragma once
#include <torch/script.h>
#include <opencv2/opencv.hpp>





torch::Tensor preprocess_image(const cv::Mat& img);

torch::Tensor run_inference(
    torch::jit::script::Module& model,
    const cv::Mat& img
);
