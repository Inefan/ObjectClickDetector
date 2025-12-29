

#include "detection.h"
#include <fstream>
#include "model_utils.h"
#include <iostream>
//# Old Variant
//bool MODEL_LOADED = false;
//
//torch::jit::script::Module load_model(const std::string& path) {
//    torch::jit::script::Module model;
//
//    try {
//        model = torch::jit::load(path);
//        model.to(torch::kCPU);
//        model.eval();
//        MODEL_LOADED = true;
//        std::cout << "Model loaded!\n";
//    }
//    catch (const c10::Error& e) {
//        std::cerr << "Model loading failed: " << e.what() << "\n";
//        MODEL_LOADED = false;
//    }
//
//    return model;
//}


bool Detector::loadModel(const std::string& path)
{
        try {
            this->model_ = torch::jit::load(path);
            this->model_.to(torch::kCPU);
            this->model_.eval();
            this->model_loaded_ = true;
            std::cout << "Model loaded!\n";
            
            return true;
        }
        
        catch (const c10::Error& e) {
            std::cerr << "Model loading failed: " << e.what() << "\n";
            this->model_loaded_ = false;
            
            return false;
        }
}
torch::Tensor preprocess_image(const cv::Mat& img_original) {
    cv::Mat img;
    cv::cvtColor(img_original, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(640, 640));

    auto tensor = torch::from_blob(
        img.data, { 1, 640, 640, 3 }, torch::kByte
    ).permute({ 0, 3, 1, 2 }).to(torch::kFloat) / 255.0;

    return tensor.clone();   
}

torch::Tensor run_inference(torch::jit::script::Module& model, const cv::Mat& img) {
    auto input = preprocess_image(img);
    std::vector<torch::jit::IValue> inputs = { input };

    try {
        auto output_iv = model.forward(inputs);

        auto output_tuple = output_iv.toTuple();

        torch::Tensor  detections = output_tuple->elements()[0].toTensor();
        return detections;
    }
    catch (const c10::Error& e) {
        std::cerr << "Inference failed: " << e.what() << "\n";
        throw;
    }
}

bool Detector::loadClasses(const std::string& path)
{
    std::string line;
    this->class_names_.clear();
    std::ifstream ReadFile(path);
    if (!ReadFile.is_open())
        return false;

    while (std::getline(ReadFile, line))
    {
        class_names_.push_back(line);
    }
    for (int i = 0; i < class_names_.size(); ++i) //Only for debug
    {//Only for debug                        
        std::cout << "ID: " << i << " Name: " << class_names_[i] << std::endl;//Only for debug
    };//Only for debug
    if (class_names_.empty()) return false;
    return true;

}