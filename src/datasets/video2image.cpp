#include <iostream>
#include <ctype.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using std::cin;
using std::cout;
using std::endl;
using std::string;

void video_to_image(string input_file_path, string output_dirc_path){
    cv::VideoCapture capture(input_file_path);
    if(!capture.isOpened()){
        cout << "Error: can not open video file." << endl;
        cout << "Please check input video file path." << endl;
        exit(1);
    }
    int count = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fps = (int)capture.get(CV_CAP_PROP_FPS);
    cout << "Total frame number: " << count << endl;

    cv::Mat frame;
    int interval = 30; //interval at which images are saved.
    int frame_num = 0;
    string output_file_path = "";

    for(int i=0;i<4*fps;i++){
        capture >> frame;
        frame_num++;
    }

    while(true){
        capture >> frame;
        if (frame.empty()){
            break;
        }
        frame_num++;

        if(frame_num % interval == 0){
            output_file_path = output_dirc_path + std::to_string(frame_num) + ".png";
            cv::imwrite(output_file_path, frame);
            cout << "Saved image(frame number: " << frame_num << ")" << endl;
            break;
        }
    }

    cout << "Converted!" << endl;
}

int main(int argc, char **argv) {
    string input_file_path, output_dirc_path;
    cout << "Input video file path: ";
    cin >> input_file_path;
    cout << "\nOutput directory path: ";
    cin >> output_dirc_path;
    cout << endl;

    video_to_image(input_file_path, output_dirc_path);

    return 0;
}
