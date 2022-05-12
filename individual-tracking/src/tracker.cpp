#include "tracker.hpp"
#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <tuple>
#include <vector>
#include "tracking_config.hpp"
#include "utils.hpp"

using cv::Mat;
using std::cout;
using std::deque;
using std::endl;
using std::map;
using std::string;
using std::tuple;
using std::vector;

typedef cv::Point2f Pixel;
typedef tuple<vector<int>, vector<float>, vector<float>, vector<float>,
              vector<vector<int>>>
    StatsResultTuple;

Tracker::Tracker(map<string, string> cfg) {
  // caputure the video.
  // if the video can not be open, it will end.
  capture = *(new cv::VideoCapture(cfg["video_path"]));
  if (!capture.isOpened()) {
    cout << "ERROR: can not open file (input video). please check file path."
         << endl;
    cout << "input path: " << cfg["video_path"] << endl;
    exit(1);
  }

  // set path
  video_path = cfg["video_path"];
  coord_dir = cfg["coord_dirctory"];
  output_stats_dir = cfg["output_stats_dirctory"];
  output_video_path = cfg["output_video_path"];

  // set tracing parameter
  load_detected_point_interval = std::stoi(cfg["load_detected_point_interval"]);
  tracking_thresh = std::stoi(cfg["tracking_thresh"]);
  template_thresh = std::stod(cfg["template_thresh"]);

  // set video infomation
  width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
  height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  total_frame = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
  fourcc = (int)capture.get(cv::CAP_PROP_FOURCC);
  fps = (double)capture.get(cv::CAP_PROP_FPS);
}

Mat Tracker::padding(Mat& frame, int padding_size) {
  /** apply zero padding to the image.
   *
   * input:
   *   frame: raw image
   *   padding_size: padding size on one side
   * output:
   *   zero padded frame
   **/

  int pad_width = frame.size().width + padding_size * 2;
  int pad_height = frame.size().height + padding_size * 2;
  Mat pad_frame = Mat::zeros(cv::Size(pad_width, pad_height), CV_8UC3);
  cv::Rect frame_region(padding_size, padding_size, frame.size().width,
                        frame.size().height);
  cv::add(pad_frame(frame_region), frame, pad_frame(frame_region));

  return pad_frame;
}

vector<int> Tracker::local_template_matching(
    Mat& prev_frame, Mat& current_frame, vector<int>& feature_vec,
    int template_size, int search_width, float matching_thresh) {
  /** create local images and perform template matching.
   * ref: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
   *
   * input:
   *   prev_frame      : original frame of template
   *   current_frame   :　frame to be searched
   *   feature_coord   : coordinate of target (center of template)
   *   template_size   : template width and height (template is square image)
   *   search_width    : serch width in four directions (± x, ± y)
   *   matching_thresh : value range is 0.0-1.0
   *
   * output: (the coordinate system is that of the original image)
   *   the matching template is
   *       found     -> center coordinate of the matched template
   *       NOT found -> empty vector
   **/

  vector<int> matching_coord;

  // check feature points that exist in the analysis area
  if (feature_vec.at(0) < 0 || feature_vec.at(1) < 0) {
    cout << "******* ERROR *******" << endl;
    cout << "x= " << feature_vec.at(0) << endl;
    cout << "y= " << feature_vec.at(1) << endl;
    cout << "*********************" << endl;
    return matching_coord;
  }

  // apply padding to the current frame
  int template_half = (int)template_size / 2;
  int padding_size = template_half + search_width;
  Mat padding_prev_frame = padding(prev_frame, padding_size);
  Mat padding_current_frame = padding(current_frame, padding_size);

  // get coordinates of feature points on padding frame
  int padding_feature_x = feature_vec.at(0) + padding_size;
  int padding_feature_y = feature_vec.at(1) + padding_size;

  // get template image
  cv::Rect template_region(padding_feature_x - template_half,
                           padding_feature_y - template_half, template_size,
                           template_size);
  Mat template_img = padding_prev_frame(template_region);

  // local image of search range
  int local_width_half = template_half + search_width;
  cv::Rect local_region(padding_feature_x - local_width_half,
                        padding_feature_y - local_width_half,
                        local_width_half * 2, local_width_half * 2);
  Mat local_img = padding_current_frame(local_region);

  // execute template matching
  Mat result;
  cv::matchTemplate(local_img, template_img, result, cv::TM_CCOEFF_NORMED);
  cv::Point max_match_point;
  double max_match_score;
  cv::minMaxLoc(result, NULL, &max_match_score, NULL, &max_match_point);
  if (max_match_score > matching_thresh) {
    // convert coordinate: left top -> center
    cv::Point match_center_coord;
    match_center_coord.x = max_match_point.x + template_half;
    match_center_coord.y = max_match_point.y + template_half;
    // convert the coordinate system to that of the input frame
    cv::Point local_frame_left_top;
    local_frame_left_top.x = padding_feature_x - local_width_half;
    local_frame_left_top.y = padding_feature_y - local_width_half;
    matching_coord.push_back(local_frame_left_top.x + match_center_coord.x -
                             padding_size);
    matching_coord.push_back(local_frame_left_top.y + match_center_coord.y -
                             padding_size);
  }

  return matching_coord;
}

StatsResultTuple Tracker::tracking() {
  /** tracking is performed based on the location of detected individuals and
   * statistics are calculated.
   **/

  // initialize processing
  int coord_index = 0;
  int frame_num = 0;
  int vector_buffer_size = 10;
  Mat prev_frame, current_frame, feature_point_frame;
  Mat tracking_mask = Mat::zeros(cv::Size(width, height), CV_8UC3);
  vector<int> current_matching_coord;
  vector<vector<int>> current_matching_coord_vec;
  vector<int> movement;
  vector<vector<int>> movement_vec;
  vector<float> movement_norm_vec;

  // set first frame and coordinate
  capture >> prev_frame;
  vector<vector<int>> prev_coord;
  string first_coord_path = coord_dir + std::to_string(coord_index) + ".csv";
  read_csv(first_coord_path, prev_coord);

  // record whether matching point was found or not (True or False)
  vector<bool> record_matching_vec;
  record_matching_vec.resize(prev_coord.size(), true);
  vector<vector<int>> start_point_vec;
  start_point_vec = prev_coord;

  // angle of direction per every frame
  vector<int> start_point(2, 0);
  vector<int> end_point(2, 0);
  int angle = 0;
  vector<int> frame_angle_vec;
  // save angle of 1sec
  vector<vector<int>> angle_vec;
  angle_vec.reserve(total_frame + vector_buffer_size);

  // statistic value
  float movement_mean, movement_var, movement_max;
  int window_size = (int)fps;
  // window size is the amount to hold statistics value
  deque<float> mean_window_deq(window_size - 1, 0.0),
      var_window_deq(window_size - 1, 0.0),
      max_window_deq(window_size - 1, 0.0);
  vector<float> mean_vec, var_vec, max_vec;
  mean_vec.reserve(total_frame + vector_buffer_size);
  var_vec.reserve(total_frame + vector_buffer_size);
  max_vec.reserve(total_frame + vector_buffer_size);

  // frame number
  vector<int> frame_vec;
  frame_vec.reserve(total_frame + vector_buffer_size);

  // save video or not
  cv::VideoWriter writer;
  if (!output_video_path.empty()) {
    writer = cv::VideoWriter(output_video_path, fourcc, fps,
                             cv::Size(width, height), true);
  }

  // tracking
  cout << "\n*************** [Start]: Tracking ***************" << endl;
  while (true) {
    capture >> current_frame;
    if (current_frame.empty()) {
      break;
    }

    frame_num++;
    frame_vec.push_back(frame_num);
    feature_point_frame = current_frame.clone();
    vector<vector<int>>().swap(current_matching_coord_vec);
    vector<vector<int>>().swap(movement_vec);
    for (int idx = 0; idx < prev_coord.size(); ++idx) {
      if (record_matching_vec.at(idx)) {
        int prev_coord_x = prev_coord.at(idx).at(0);
        int prev_coord_y = prev_coord.at(idx).at(1);
        current_matching_coord = local_template_matching(
            prev_frame, current_frame, prev_coord.at(idx), 50, 30,
            template_thresh);
        if (!current_matching_coord.empty()) {
          vector<int>().swap(movement);
          int matching_x = current_matching_coord.at(0);
          int matching_y = current_matching_coord.at(1);
          cv::line(tracking_mask, cv::Point(prev_coord_x, prev_coord_y),
                   cv::Point(matching_x, matching_y), cv::Scalar(0, 0, 255), 2,
                   cv::LINE_AA);
          current_matching_coord_vec.push_back(current_matching_coord);
          movement.push_back(std::abs(prev_coord_x - matching_x));
          movement.push_back(std::abs(prev_coord_y - matching_y));
          movement_vec.push_back(movement);

          // plot feature point
          cv::circle(feature_point_frame, cv::Point(matching_x, matching_y), 3,
                     cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
          // put individual ID
          cv::putText(feature_point_frame, std::to_string(idx),
                      cv::Point(matching_x, matching_y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1,
                      CV_AA);

          // update record of matching and previous coordinate
          record_matching_vec.at(idx) = true;
          prev_coord.at(idx) = current_matching_coord;
        }
      } else {
        record_matching_vec.at(idx) = false;
      }
    }

    // calculate statistic value
    if (!movement_vec.empty()) {
      movement_norm_vec = point_norm(movement_vec);
      movement_mean = std::accumulate(movement_norm_vec.begin(),
                                      movement_norm_vec.end(), 0.0) /
                      movement_norm_vec.size();
      movement_var = vector_variance(movement_norm_vec, movement_mean);
      movement_max =
          *std::max_element(movement_norm_vec.begin(), movement_norm_vec.end());
    } else {
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
    mean_vec.push_back(
        std::accumulate(mean_window_deq.begin(), mean_window_deq.end(), 0.0));
    var_vec.push_back(
        std::accumulate(var_window_deq.begin(), var_window_deq.end(), 0.0));
    max_vec.push_back(
        std::accumulate(max_window_deq.begin(), max_window_deq.end(), 0.0));

    if (writer.isOpened()) {
      // add tracking line
      cv::add(feature_point_frame, tracking_mask, feature_point_frame);
      writer << feature_point_frame;
    }

    // update
    mean_window_deq.pop_front();
    var_window_deq.pop_front();
    max_window_deq.pop_front();
    prev_frame = current_frame.clone();
    if (frame_num % load_detected_point_interval == 0) {
      // debug
      break;

      // reset tracking mask
      tracking_mask = Mat::zeros(cv::Size(width, height), CV_8UC3);

      // calculate angle of direction
      for (int i = 0; i < record_matching_vec.size(); ++i) {
        if (record_matching_vec.at(i)) {
          angle = calc_angle(start_point_vec.at(i), prev_coord.at(i));
          frame_angle_vec.push_back(angle);
        }
      }

      // fill the sikp width with the same value
      for (int i = 0; i < load_detected_point_interval; ++i) {
        angle_vec.push_back(frame_angle_vec);
      }

      // release memory of vector of angle
      vector<int>().swap(frame_angle_vec);

      // load the detected coordinates of the next frame
      coord_index += load_detected_point_interval;
      string next_coord_path = coord_dir + std::to_string(coord_index) + ".csv";
      std::ifstream ifs(next_coord_path);
      if (ifs.is_open()) {
        vector<vector<int>>().swap(prev_coord);
        read_csv(next_coord_path, prev_coord);
      }

      // reset matching record
      vector<bool>().swap(record_matching_vec);
      record_matching_vec.resize(prev_coord.size(), true);
      vector<vector<int>>().swap(start_point_vec);
      start_point_vec = prev_coord;
    }
  }

  // termination processing
  cv::destroyAllWindows();
  if (writer.isOpened()) {
    cout << "The tracking video was saved in " << output_video_path << endl;
    writer.release();
  }
  cout << "\n*************** [End]: Tracking ***************" << endl;

  // combine computed statistics into a single tuple
  StatsResultTuple tracking_stats_tuple =
      std::tie(frame_vec, mean_vec, var_vec, max_vec, angle_vec);

  return tracking_stats_tuple;
}

void Tracker::save_stats_results(StatsResultTuple& status_results) {
  /** save statistics computed by tracking
   *
   **/
  int tuple_size = std::tuple_size<StatsResultTuple>::value;
  assert(tuple_size == 5 &&
         "The tuple size of the statistical results is expected to be 5.");

  // extract each result vector
  auto frame_vec = std::get<0>(status_results);
  auto mean_vec = std::get<1>(status_results);
  auto var_vec = std::get<2>(status_results);
  auto max_vec = std::get<3>(status_results);
  auto angle_vec = std::get<4>(status_results);

  cout << "\n*******************************************" << endl;
  // mean result
  vector<string> mean_header{"frame_num", "mean"};
  string save_mean_path = output_stats_dir + "mean.csv";
  write_csv(mean_vec, frame_vec, mean_header, save_mean_path);

  // variance result
  vector<string> var_header{"frame_num", "var"};
  string save_var_path = output_stats_dir + "var.csv";
  write_csv(var_vec, frame_vec, var_header, save_var_path);

  // max result
  vector<string> max_header{"frame_num", "max"};
  string save_max_path = output_stats_dir + "max.csv";
  write_csv(max_vec, frame_vec, max_header, save_max_path);

  // angle result
  vector<string> angle_header{"frame_num", "angle"};
  string save_angle_path = output_stats_dir + "angle.csv";
  write_csv_2d(angle_vec, frame_vec, angle_header, save_angle_path);
  cout << "*******************************************" << endl;
}