#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <ctype.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

void movie_to_image(std::string inputFilePath, std::string outputDircPath){
    cv::VideoCapture capture(inputFilePath);
    if(!capture.isOpened()){
        cout << "Error: can not open movie file." << endl;
        cout << "Please check input movie file path." << endl;
        exit(1);
    }
    int count = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fps = (int)capture.get(CV_CAP_PROP_FPS);
    cout << "Total frame number: " << count << endl;
    cout << "Input movie fps: " << fps << endl;

    cv::Mat frame;
    int interval = fps * 60; //interval at which images are saved.
    int frameNum = 0;
    string outputFilePath = "";

    while(true){
        capture >> frame;
        if (frame.empty()){
            break;
        }
        frameNum++;

        if(frameNum % interval == 0){
            outputFilePath = outputDircPath + "/13_" + to_string(frameNum) + ".png";
            cv::imwrite(outputFilePath, frame);
            cout << "Saved image(frame number: " << frameNum << ")" << endl;
        }
    }

    cout << "Converted!" << endl;
}

int main(int argc, char **argv) {
    string inputFilePath, outputDircPath;
    cout << "Input movie file path: ";
    cin >> inputFilePath;
    cout << "\nOutput directory path: ";
    cin >> outputDircPath;
    cout << endl;

    movie_to_image(inputFilePath, outputDircPath);

    return 0;
}
