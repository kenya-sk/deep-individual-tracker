#include <ctype.h>
#include <dirent.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include "timer.hpp"
#include "tracking_config.hpp"

using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::string;

extern void tracking(string, string, string, string);

int main(int argc, char** argv) {
  TrackingConfig trackingConfig;
  string config_path = "../conf/tracking_config.cfg";
  map<string, string> cfg = trackingConfig.config_parser(config_path);

  // set timer
  Timer timer;

  tracking(cfg["video_path"], cfg["coord_dirctory"],
           cfg["output_stats_dirctory"], cfg["output_video_path"]);

  // display calculation time
  timer.output_calculate_time();

  return 0;
}