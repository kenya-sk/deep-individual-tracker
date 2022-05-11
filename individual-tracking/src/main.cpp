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

  // start timer
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  tracking(cfg["video_path"], cfg["coord_dirctory"],
           cfg["output_stats_dirctory"], cfg["output_video_path"]);

  // display calculation time
  end = std::chrono::system_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  cout << "elapsed time: " << elapsed << "sec." << endl;

  return 0;
}