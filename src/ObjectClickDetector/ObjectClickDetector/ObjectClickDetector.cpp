// ObjectClickDetector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        cout << "Click at (" << x << ", " << y << ")" << endl;
    }
}

int main() {
    string imagePath = "C:\\Users\\MS\\OneDrive\\Desktop\\ObjectClickDetector\\test.jpg";
    cv::Mat img = imread(imagePath);

    if (img.empty()) {
        cout << "Could not open or find the image!\n";
        return -1;
    }

    namedWindow("Image", WINDOW_AUTOSIZE);
    setMouseCallback("Image", onMouse);
    imshow("Image", img);

    cout << "Click on the image window.\n";
    waitKey(0);
    return 0;
}
torch::Tensor preprocess_image(const cv::Mat& img_original) {
    const int H = 640;
    const int W = 640;
    cv::Mat img;

    cv::cvtColor(img_original, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(W, H));

    torch::Tensor tensor = torch::from_blob(
        img.data,
        { 1, H, W, 3 },
        torch::kByte 
    );
    tensor = tensor.permute({ 0,3,1,2 });

    tensor = tensor.to(torch::kFloat);

    tensor = tensor.div(255.0);

    return tensor;
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
