#pragma once

#include <chrono>

typedef std::chrono::system_clock::time_point time_point;

// class for managing computation time
class Timer {
 public:
  time_point start_time;

  Timer();
  void output_calculate_time();
};