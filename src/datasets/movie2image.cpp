#include <iostream>
#include <ctype.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

void movie_to_image(std::string input_file_path, std::string output_dirc_path){
    cv::VideoCapture capture(input_file_path);
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
    int frame_num = 0;
    string output_file_path = "";

    while(true){
        capture >> frame;
        if (frame.empty()){
            break;
        }
        frame_num++;

        if(frame_num % interval == 0){
            output_file_path = output_dirc_path + to_string(frame_num) + ".png";
            cv::imwrite(output_file_path, frame);
            cout << "Saved image(frame number: " << frame_num << ")" << endl;
        }
    }

    cout << "Converted!" << endl;
}

int main(int argc, char **argv) {
    string input_file_path, output_dirc_path;
    cout << "Input movie file path: ";
    cin >> input_file_path;
    cout << "\nOutput directory path: ";
    cin >> output_dirc_path;
    cout << endl;

    movie_to_image(input_file_path, output_dirc_path);

    return 0;
}
