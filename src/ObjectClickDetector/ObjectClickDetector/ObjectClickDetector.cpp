// ObjectClickDetector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "detection.h"
#include "model_utils.h"
#include "mouse_handler.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;




int main() {
    // 1) Завантажуєм модель!
    auto model = load_model("C:\\Users\\MS\\OneDrive\\Desktop\\ObjectClickDetector\\model\\yolov5s.torchscript");
    //2) Завантажуєм фото!
    string image_path = ("C:\\Users\\MS\\OneDrive\\Desktop\\ObjectClickDetector\\photos\\test.jpg");
    original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cout << "Image not found!\n";
        return -1;
    }
    // 3) Інференс
    auto output = run_inference(model, original_image);
    cout << "Got output, continuing UI\n";
    // 4) Парс Yolo
    parse_yolo_output(output, original_image.rows, original_image.cols);
    // 5)  Малювання боксів
    cv::Mat display = original_image.clone();
    draw_boxes(display);
    // 6) Додаємо ОнМоус
    cv::namedWindow("Image");
    cv::setMouseCallback("Image", onMouse);
    cv::imshow("Image", display);

    cout << "Click on an object.\n";
    cv::waitKey(0);
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
