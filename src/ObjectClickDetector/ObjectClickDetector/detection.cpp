#include "detection.h"
#include "model_utils.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
void Detector::detect(const cv::Mat& image)
{
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    boxes_.clear();

    torch::Tensor output = run_inference(model_, image);

    torch::Tensor pred = output.squeeze(0);

    const float conf_threshold = 0.4f;
    const float iou_threshold  = 0.45f;

    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<int> classes;

    for (int i = 0; i < pred.size(0); ++i) {
        auto row = pred[i];

        float obj_conf = row[4].item<float>();
        if (obj_conf < conf_threshold) continue;

        auto cls_scores = row.slice(0, 5, row.size(0));
        int cls = cls_scores.argmax().item<int>();
        float cls_conf = cls_scores[cls].item<float>();

        float score = obj_conf * cls_conf;
        if (score < conf_threshold) continue;

        float xc = row[0].item<float>();
        float yc = row[1].item<float>();
        float w  = row[2].item<float>();
        float h  = row[3].item<float>();

        int x = int((xc - w / 2) * image.cols / 640);
        int y = int((yc - h / 2) * image.rows / 640);
        int ww = int(w * image.cols / 640);
        int hh = int(h * image.rows / 640);

        rects.emplace_back(x, y, ww, hh);
        scores.push_back(score);
        classes.push_back(cls);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(rects, scores, conf_threshold, iou_threshold, indices);

    for (int idx : indices) {
        boxes_.push_back({ rects[idx], classes[idx], scores[idx] });
    }
}

void Detector::draw(cv::Mat& img) const {
    for (const auto& box : boxes_) {
        cv::rectangle(img, box.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "class " + std::to_string(box.cls);
        cv::putText(img, label,
            cv::Point(box.box.x, box.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 255, 0), 2);
    }
}


int Detector::getBoxIndex(int x, int y) const
{
    for (int i = 0; i < boxes_.size(); i++) {
        if (boxes_[i].box.contains(cv::Point(x, y))) {
            return i;
        }
    }
    return -1;
}
