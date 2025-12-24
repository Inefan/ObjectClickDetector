#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "detect_types.h"
#include <torch/script.h>
#include <torch/torch.h>

class Detector {
public:
    bool loadModel(const std::string& path);
    void detect(const cv::Mat& image);
    void draw(cv::Mat& img) const;
    int getBoxIndex(int x, int y) const;
    bool loadClasses(const std::string& path);
    


private:
    torch::jit::script::Module model_;
    std::vector<DetectBox> boxes_;
    bool model_loaded_ = false;
    std::vector<std::string> class_names_;
};
