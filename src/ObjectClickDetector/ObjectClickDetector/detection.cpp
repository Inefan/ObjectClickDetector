#include "detection.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>

void parse_yolo_output(const torch::Tensor& output, int orig_h, int orig_w)
{
    detected_boxes.clear();

    torch::Tensor pred = output.squeeze();

    float conf_threshold = 0.4f;
    float iou_threshold = 0.45f;

    std::vector<DetectBox> boxes;

    for (int i = 0; i < pred.size(0); i++)
    {
        auto row = pred[i];

        float obj_conf = row[4].item<float>();
        if (obj_conf < conf_threshold)
            continue;

        auto class_score = row.slice(0, 5, row.size(0));
        auto max_class = class_score.argmax().item<int>();
        float class_conf = class_score[max_class].item<float>();

        float score = obj_conf * class_conf;
        if (score < conf_threshold)
            continue;

        float xc = row[0].item<float>();
        float yc = row[1].item<float>();
        float w = row[2].item<float>();
        float h = row[3].item<float>();

        int x1 = int((xc - w / 2.f) * orig_w / 640.f);
        int y1 = int((yc - h / 2.f) * orig_h / 640.f);
        int x2 = int((xc + w / 2.f) * orig_w / 640.f);
        int y2 = int((yc + h / 2.f) * orig_h / 640.f);

        boxes.push_back({
            cv::Rect(x1, y1, x2 - x1, y2 - y1),
            max_class,
            score
        });

    }
    std::vector<cv::Rect> rects;
    std::vector<float> scores;

    for (auto& b : boxes) {
        rects.push_back(b.box);
        scores.push_back(b.score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(rects, scores, conf_threshold, iou_threshold, indices);

    for (int ind : indices) 
        detected_boxes.push_back(boxes[ind]);
        
    std::cout << "detect boxes!" << detected_boxes.size() << " objects\n";
     
}

void draw_boxes(cv::Mat& img)
{
    for (auto& b : detected_boxes) {
        cv::rectangle(img, b.box, cv::Scalar( 0,255,0 ), 2);
    }
}


