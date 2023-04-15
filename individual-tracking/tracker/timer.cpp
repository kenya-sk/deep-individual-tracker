#include "timer.hpp"

#include <chrono>
#include <iostream>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::hours;
using std::chrono::minutes;
using std::chrono::seconds;
using std::chrono::system_clock;

Timer::Timer() { start_time = system_clock::now(); }

void Timer::output_calculate_time() {
  time_point end_time = system_clock::now();
  int elapsed_hours = duration_cast<hours>(end_time - start_time).count();
  int elapsed_minutes =
      duration_cast<minutes>((end_time - start_time)).count() -
      elapsed_hours * 60;
  int elapsed_seconds = duration_cast<seconds>(end_time - start_time).count() -
                        elapsed_hours * 3600 - elapsed_minutes * 60;
  cout << "elapsed time: " << elapsed_hours << "[h] " << elapsed_minutes
       << "[min] " << elapsed_seconds << "[sec]" << endl;
}