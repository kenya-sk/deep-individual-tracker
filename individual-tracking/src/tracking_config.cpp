#include "tracking_config.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>

using std::cerr;
using std::map;
using std::string;

TrackingConfig::TrackingConfig() {
  cfg = {{"video_path", ""},
         {"coord_dirctory", ""},
         {"output_stats_dirctory", ""},
         {"output_video_path", ""},
         {"tracking_thresh", ""},
         {"template_thresh", ""}};
}

map<string, string> TrackingConfig::config_parser(string config_path) {
  std::ifstream cFile(config_path);
  if (cFile.is_open()) {
    string line;
    while (getline(cFile, line)) {
      // remove white space
      line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
      // skip comment line or blank line
      if (line[0] == '#' || line.empty()) continue;
      auto delimiter_pos = line.find("=");
      auto key = line.substr(0, delimiter_pos);
      auto value = line.substr(delimiter_pos + 1);

      // If the key exists in the config member, register the value.
      if (cfg.count(key)) {
        cfg[key] = value;
      }
    }
  } else {
    cerr << "Couldn't open config file.\n";
  }

  return cfg;
}