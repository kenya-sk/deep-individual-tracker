#include <iostream>
#include <chrono>
#include <numeric>
#include <vector>
#include <deque>
#include <algorithm>
#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <ctype.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using std::cin;
using std::cout;
using std::endl;
using std::string;

extern void tracking(string, string, string, string);

int main(int argc, char **argv)
{
    string video_filepath;
    string cord_dircpath;
    string output_stats_dirc;
    string is_saved = "0"; // default: NOT SAVE output video
    string output_video_path = "";

    // input data path from stdin
    cout << "input video path: ";
    cin >> video_filepath;
    cout << "input cordinate path: ";
    cin >> cord_dircpath;
    cout << "input the output statistics directory: ";
    cin >> output_stats_dirc;
    cout << "save tracking video (0:NO, 1:YES)";
    cin >> is_saved;
    if (stoi(is_saved))
    {
        cout << "input the output video path: ";
        cin >> output_video_path;
    }
    else
    {
        cout << "output video is NOT save." << endl;
    }

    // start timer
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    tracking(video_filepath, cord_dircpath, output_stats_dirc, output_video_path);

    // display calculation time
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    cout << "elapsed time: " << elapsed << "sec." << endl;

    return 0;
}