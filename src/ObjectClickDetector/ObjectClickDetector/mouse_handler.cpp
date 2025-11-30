#include "mouse_handler.h"
#include <iostream>

using namespace std;

cv::Mat original_image;
int selected_box = -1;
std::vector<DetectBox> detected_boxes;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }
    cout << "Click at (" << x << ", " << y << ")" << endl;

    selected_box = -1;

    for (int i = 0; i < detected_boxes.size(); i++) {
        const auto& b = detected_boxes[i].box;

        int x1 = b.x;
        int y1 = b.y;
        int x2 = b.x + b.width;
        int y2 = b.y + b.height;

        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
            selected_box = i;
            break;
        }
    }
    if (selected_box == -1) {
        cout << "Clicked outsite any box" << endl;
        return;
    }
    cv::Mat img = original_image.clone();
    cv::rectangle(img, detected_boxes[selected_box].box, cv::Scalar(0, 255, 0), 2);

    string label = "class" + std::to_string(detected_boxes[selected_box].cls);
    cv::putText(img, label, cv::Point(x + 5, y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Image", img);

}
