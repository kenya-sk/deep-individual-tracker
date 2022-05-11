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
//#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include "utils.hpp"

using cv::Mat;
using std::cout;
using std::deque;
using std::endl;
using std::string;
using std::vector;

typedef cv::Point2f Pixel;

// define anti-aliasing
// int CV_AA = 16;

Mat padding(Mat &frame, int padding_size)
{
    /**
     * input:
     *   frame: original image
     *   padding_size: padding width
     * output:
     *   zero padding frame
     **/

    int pad_width = frame.size().width + padding_size * 2;
    int pad_height = frame.size().height + padding_size * 2;
    Mat pad_frame = Mat::zeros(cv::Size(pad_width, pad_height), CV_8UC3);
    cv::Rect frame_region(padding_size, padding_size, frame.size().width, frame.size().height);
    cv::add(pad_frame(frame_region), frame, pad_frame(frame_region));

    return pad_frame;
}

vector<int> local_match_template(Mat &prev_frame, Mat &current_frame, vector<int> &feature_vec, int template_size, int search_width, float matching_thresh)
{
    /**
     * create local images and perform template matching
     * input:
     *   prev_frame      : original frame of template
     *   current_frame   :　frame to be searched
     *   feature_cord    : coordinate of target (center of template)
     *   template_size   : template width and height (template is square image)
     *   search_width    : serch width in four directions (± x, ± y)
     *   matching_thresh : value range is 0.0-1.0
     * output: (the coordinate system is that of the original image)
     *   the matching template is
     *       found     -> center coordinate of the matched template
     *       NOT found -> empty vector
     **/

    vector<int> matching_cord;

    // check feature point in analysis area
    if (feature_vec.at(0) < 0 || feature_vec.at(1) < 0)
    {
        cout << "******* ERROR *******" << endl;
        cout << "x= " << feature_vec.at(0) << endl;
        cout << "y= " << feature_vec.at(1) << endl;
        cout << "*********************" << endl;
        return matching_cord;
    }

    // padding frame
    int template_half = (int)template_size / 2;
    int padding_size = template_half + search_width;
    Mat padding_prev_frame = padding(prev_frame, padding_size);
    Mat padding_current_frame = padding(current_frame, padding_size);

    // feature point cord of padding image
    int padding_feature_x = feature_vec.at(0) + padding_size;
    int padding_feature_y = feature_vec.at(1) + padding_size;

    // template image
    cv::Rect template_region(padding_feature_x - template_half, padding_feature_y - template_half, template_size, template_size);
    Mat template_img = padding_prev_frame(template_region);

    // local image of search range
    int local_width_half = template_half + search_width;
    cv::Rect local_region(padding_feature_x - local_width_half, padding_feature_y - local_width_half, local_width_half * 2, local_width_half * 2);
    Mat local_img = padding_current_frame(local_region);

    // template matching
    Mat result;
    cv::matchTemplate(local_img, template_img, result, cv::TM_CCOEFF_NORMED);
    cv::Point max_match_point;
    double max_match_score;
    cv::minMaxLoc(result, NULL, &max_match_score, NULL, &max_match_point);
    if (max_match_score > matching_thresh)
    {
        // fix cord: left top -> center
        cv::Point match_center_cord;
        match_center_cord.x = max_match_point.x + template_half;
        match_center_cord.y = max_match_point.y + template_half;
        // fix to coordinate system of the input frame
        cv::Point local_frame_left_top;
        local_frame_left_top.x = padding_feature_x - local_width_half;
        local_frame_left_top.y = padding_feature_y - local_width_half;
        matching_cord.push_back(local_frame_left_top.x + match_center_cord.x - padding_size);
        matching_cord.push_back(local_frame_left_top.y + match_center_cord.y - padding_size);
    }

    return matching_cord;
}

void tracking(string video_filepath, string cord_dircpath, string output_stats_dirc, string output_video_path)
{
    /**
     * The feature points are handled based on the Hungarian algorithm.
     *
     * input:
     *   video_filepath: video file path corresponding to cord_dircpath.
     *   cord_dircpath: cordinate of feature (.csv).
     *   output_stats_dirc: save statistics value (mean, var, max)
     *   output_video_path: output video file path
     *
     * output: video of tracking feature points.
     **/

    // caputure the video.
    // if the video can not be read, it will end.
    cv::VideoCapture capture(video_filepath);
    if (!capture.isOpened())
    {
        cout << "ERROR: can not open file (input video). please check file path." << endl;
        cout << "input path: " << video_filepath << endl;
        exit(1);
    }

    // video infomation
    int width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int total_frame = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
    int fourcc = (int)capture.get(cv::CAP_PROP_FOURCC);
    double fps = (double)capture.get(cv::CAP_PROP_FPS);
    display_video_info(video_filepath, width, height, total_frame, fourcc, fps);

    // initialize processing
    int skip = 30; // skip number of frame
    int tracking_thresh = 15;
    int cord_index = 0;
    int frame_num = 0;
    int vector_buffer_size = 10;
    float template_thresh = 0.7;
    Mat prev_frame, current_frame, feature_point_frame;
    Mat tracking_mask = Mat::zeros(cv::Size(width, height), CV_8UC3);

    capture >> prev_frame;
    vector<vector<int>> prev_cord;
    read_csv(cord_dircpath + std::to_string(cord_index) + ".csv", prev_cord, ',');
    vector<int> current_matching_cord;
    vector<vector<int>> current_matching_cord_vec;
    vector<int> movement;
    vector<vector<int>> movement_vec;
    vector<float> movement_norm_vec;

    // record whether matching point was found or not (True or False)
    vector<bool> record_matching_vec;
    record_matching_vec.resize(prev_cord.size(), true);
    vector<vector<int>> start_point_vec;
    start_point_vec = prev_cord;

    // dgree of direction (every frame)
    vector<int> start_point(2, 0);
    vector<int> end_point(2, 0);
    int degree = 0;
    vector<int> frame_degree_vec;
    // save degree of 1sec
    vector<vector<int>> degree_vec;
    degree_vec.reserve(total_frame + vector_buffer_size);

    // statistic value
    float movement_mean, movement_var, movement_max;
    int window_size = (int)fps; // window size is the amount to hold statistics value
    deque<float> mean_window_deq(window_size - 1, 0.0), var_window_deq(window_size - 1, 0.0), max_window_deq(window_size - 1, 0.0);
    vector<float> mean_vec, var_vec, max_vec;
    mean_vec.reserve(total_frame + vector_buffer_size);
    var_vec.reserve(total_frame + vector_buffer_size);
    max_vec.reserve(total_frame + vector_buffer_size);

    // frame number
    vector<int> frame_vec;
    frame_vec.reserve(total_frame + vector_buffer_size);

    // save video or not
    cv::VideoWriter writer;
    if (!output_video_path.empty())
    {
        writer = cv::VideoWriter(output_video_path, fourcc, fps, cv::Size(width, height), true);
    }

    // start timer
    std::chrono::system_clock::time_point start_time;
    start_time = std::chrono::system_clock::now();

    // tracking
    cout << "\n*******************************************" << endl;
    cout << "START: tracking" << endl;
    while (true)
    {
        capture >> current_frame;
        if (current_frame.empty())
        {
            break;
        }

        frame_num++;
        frame_vec.push_back(frame_num);
        if (frame_num % (total_frame / 100) == 0)
        {
            double elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start_time).count();
            cout << "NOW: " << frame_num << '/' << total_frame << "  elapsed time: " << elapsed_time << endl;
        }

        feature_point_frame = current_frame.clone();
        std::vector<std::vector<int>>().swap(current_matching_cord_vec);
        std::vector<std::vector<int>>().swap(movement_vec);
        for (int idx = 0; idx < prev_cord.size(); ++idx)
        {
            if (record_matching_vec.at(idx))
            {
                int prev_cord_x = prev_cord.at(idx).at(0);
                int prev_cord_y = prev_cord.at(idx).at(1);
                current_matching_cord = local_match_template(prev_frame, current_frame, prev_cord.at(idx), 50, 30, template_thresh);
                if (!current_matching_cord.empty())
                {
                    std::vector<int>().swap(movement);
                    int matching_x = current_matching_cord.at(0);
                    int matching_y = current_matching_cord.at(1);
                    cv::line(tracking_mask, cv::Point(prev_cord_x, prev_cord_y), cv::Point(matching_x, matching_y), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                    current_matching_cord_vec.push_back(current_matching_cord);
                    movement.push_back(std::abs(prev_cord_x - matching_x));
                    movement.push_back(std::abs(prev_cord_y - matching_y));
                    movement_vec.push_back(movement);

                    // plot feature point
                    cv::circle(feature_point_frame, cv::Point(matching_x, matching_y), 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                    // put individual ID
                    cv::putText(feature_point_frame, std::to_string(idx), cv::Point(matching_x, matching_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, CV_AA);

                    // update record of matching and previous coordinate
                    record_matching_vec.at(idx) = true;
                    prev_cord.at(idx) = current_matching_cord;
                }
            }
            else
            {
                record_matching_vec.at(idx) = false;
            }
        }

        // calculate statistic value
        if (!movement_vec.empty())
        {
            movement_norm_vec = point_norm(movement_vec);
            movement_mean = std::accumulate(movement_norm_vec.begin(), movement_norm_vec.end(), 0.0) / movement_norm_vec.size();
            movement_var = vector_variance(movement_norm_vec, movement_mean);
            movement_max = *std::max_element(movement_norm_vec.begin(), movement_norm_vec.end());
        }
        else
        {
            movement_mean = 0.0;
            movement_var = 0, 0;
            movement_max = 0.0;
        }

        // statistic value push window deque
        mean_window_deq.push_back(movement_mean);
        var_window_deq.push_back(movement_var);
        max_window_deq.push_back(movement_max);
        assert(mean_window_deq.size() == window_size);
        assert(var_window_deq.size() == window_size);
        assert(max_window_deq.size() == window_size);

        // window value push frame vector
        mean_vec.push_back(std::accumulate(mean_window_deq.begin(), mean_window_deq.end(), 0.0));
        var_vec.push_back(std::accumulate(var_window_deq.begin(), var_window_deq.end(), 0.0));
        max_vec.push_back(std::accumulate(max_window_deq.begin(), max_window_deq.end(), 0.0));

        if (writer.isOpened())
        {
            // add tracking line
            cv::add(feature_point_frame, tracking_mask, feature_point_frame);
            writer << feature_point_frame;
        }

        // update
        mean_window_deq.pop_front();
        var_window_deq.pop_front();
        max_window_deq.pop_front();
        prev_frame = current_frame.clone();
        if (frame_num % skip == 0)
        {
            // debug
            break;

            // reset tracking mask
            tracking_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

            // calculate degree of direction
            for (int i = 0; i < record_matching_vec.size(); ++i)
            {
                if (record_matching_vec.at(i))
                {
                    degree = calc_angle(start_point_vec.at(i), prev_cord.at(i));
                    frame_degree_vec.push_back(degree);
                }
            }

            // fill the sikp width with the same value
            for (int i = 0; i < skip; ++i)
            {
                degree_vec.push_back(frame_degree_vec);
            }

            // release memory of vector of degree
            std::vector<int>().swap(frame_degree_vec);

            // load coordinate of detection by CNN
            cord_index += skip;
            std::ifstream ifs(cord_dircpath + std::to_string(cord_index) + ".csv");
            if (ifs.is_open())
            {
                std::vector<std::vector<int>>().swap(prev_cord);
                read_csv(cord_dircpath + std::to_string(cord_index) + ".csv", prev_cord, ',');
            }

            // reset matching record
            std::vector<bool>().swap(record_matching_vec);
            record_matching_vec.resize(prev_cord.size(), true);
            std::vector<std::vector<int>>().swap(start_point_vec);
            start_point_vec = prev_cord;
        }
    }

    cv::destroyAllWindows();
    cout << "DONE: tracking" << endl;
    cout << "DONE: calculate statistic value" << endl;
    if (writer.isOpened())
    {
        cout << "save output video in " << output_video_path << endl;
        writer.release();
    }
    cout << "*******************************************\n"
         << endl;

    // // save statistics value
    // cout << "\n*******************************************" << endl;
    // cout << "save statistic data in csv file" << endl;
    // std::vector<string> mean_header{"frame_num", "mean"};
    // write_csv(mean_vec, frame_vec, mean_header, output_stats_dirc + "mean.csv");
    // std::vector<string> var_header{"frame_num", "var"};
    // write_csv(var_vec, frame_vec, var_header, output_stats_dirc + "var.csv");
    // std::vector<string> max_header{"frame_num", "max"};
    // write_csv(max_vec, frame_vec, max_header, output_stats_dirc + "max.csv");
    // std::vector<string> degree_header{"frame_num", "degree"};
    // write_csv_2d(degree_vec, frame_vec, degree_header, output_stats_dirc + "degree.csv");
    // cout << "*******************************************\n"
    //      << endl;
}
