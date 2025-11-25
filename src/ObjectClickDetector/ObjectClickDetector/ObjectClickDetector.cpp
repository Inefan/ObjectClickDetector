// ObjectClickDetector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;



cv::Mat original_image;
int selected_box = -1;
struct DetectBox {
    cv::Rect box;
    int cls;
    float score;
};
vector<DetectBox> detected_boxes;

void onMouse(int event, int x, int y, int flags, void* userdata){
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
torch::jit::script::Module model;
void load_model(const std::string& path) {
    try {
        model = torch::jit::load(path);
        model.to(torch::kCPU);
        model.eval();
        std::cout << "Model loaded successfully!\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
    }
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
torch::Tensor load_inference(const cv::Mat & img) {

    torch::Tensor input = preprocess_image(img);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::Tensor outputs = model.forward(inputs).toTensor();

    return outputs;

}

int main() {
    string imagePath = "C:\\Users\\MS\\OneDrive\\Desktop\\ObjectClickDetector\\test.jpg";

    load_model("C:\\Users\\MS\\OneDrive\\Desktop\\ObjectClickDetector\\model\\yolov5s.torchscript");
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
    load_inference(img);

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
