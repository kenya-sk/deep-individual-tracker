#pragma once

#include <map>

using std::map;
using std::string;

class TrackingConfig
{
public:
    map<string, string> cfg;

    TrackingConfig();
    map<string, string> config_parser(string config_path);
};