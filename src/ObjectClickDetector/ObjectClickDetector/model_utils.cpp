#include "model_utils.h"
#include <iostream>

bool MODEL_LOADED = false;

torch::jit::script::Module load_model(const std::string& path) {
    torch::jit::script::Module model;

    try {
        model = torch::jit::load(path);
        model.to(torch::kCPU);
        model.eval();
        MODEL_LOADED = true;
        std::cout << "Model loaded!\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Model loading failed: " << e.what() << "\n";
        MODEL_LOADED = false;
    }

    return model;
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

    if (!MODEL_LOADED) {
        throw std::runtime_error("ERROR: model was NOT loaded, cannot run inference");
    }

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

