#pragma once

#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <tuple>
#include <vector>

using cv::Mat;
using cv::VideoCapture;
using std::map;
using std::string;
using std::tuple;
using std::vector;

typedef tuple<vector<int>, vector<float>, vector<float>, vector<float>,
              vector<vector<int>>>
    StatsResultTuple;

class Tracker {
 public:
  string video_path;
  string coord_dir;
  string output_stats_dir;
  string output_video_path;
  VideoCapture capture;
  int width;
  int height;
  int total_frame;
  int fourcc;
  double fps;
  int load_detected_point_interval;
  int tracking_thresh;
  double template_thresh;

  Tracker(map<string, string> cfg);

  Mat padding(Mat& frame, int padding_size);

  vector<int> local_template_matching(Mat& prev_frame, Mat& current_frame,
                                      vector<int>& feature_vec,
                                      int template_size, int search_width,
                                      float matching_thresh);

  StatsResultTuple tracking();
};