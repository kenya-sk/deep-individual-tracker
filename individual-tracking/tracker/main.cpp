#include <iostream>
#include <map>

#include "timer.hpp"
#include "tracker.hpp"
#include "tracking_config.hpp"
#include "utils.hpp"

using std::cout;
using std::endl;
using std::map;
using std::string;

typedef tuple<vector<int>, vector<float>, vector<float>, vector<float>,
              vector<vector<int>>>
    StatsResultTuple;

int main(int argc, char** argv) {
  TrackingConfig trackingConfig;
  string config_path = "../config/tracking_config.cfg";
  map<string, string> cfg = trackingConfig.config_parser(config_path);

  // set timer
  Timer timer;

  // define tracker
  Tracker tracker = Tracker(cfg);
  display_video_info(tracker.video_path, tracker.width, tracker.height,
                     tracker.total_frame, tracker.fourcc, tracker.fps);

  // tracking and save results
  StatsResultTuple stats_tuple = tracker.tracking();
  tracker.save_stats_results(stats_tuple);

  // display calculation time
  timer.output_calculate_time();

  return 0;
}