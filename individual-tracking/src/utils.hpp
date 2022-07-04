#pragma once

#include <string>
#include <vector>

using std::string;
using std::vector;

// number fourcc (video codec) convert to string
string fourcc_to_number(int fourcc);

// output video information to standard output.
void display_video_info(string video_path, int width, int height,
                        int total_frame, int fourcc, double fps);

// returns True if the input character is not a number.
bool is_not_digit(char c);

// numerical compare two strings.
bool numeric_string_compare(const string& s1, const string& s2);

// make a vector of filepath with the specified extension.
void make_file_vector(string input_dircpath, vector<string>& file_vec,
                      string file_extension);

// create vector table from csv data.
void read_csv(string input_csv_file_path, vector<vector<int>>& table,
              const char delimiter = ',');

// write vector data to csv file.
void write_csv(vector<float>& data_vec, vector<int>& frame_num_vec,
               vector<string>& header_vec, string output_csv_path);

// write 2D vector data to csv file.
void write_csv_2d(vector<vector<int>>& data_vec, vector<int>& frame_num_vec,
                  vector<string>& header_vec, string output_csv_path);

// caluculate the norm between two points.
vector<float> point_norm(vector<vector<int>>& point_vec);

// caluculate the variance of value vector.
float vector_variance(vector<float>& value_vec, float mean);

// traget has moved or not.
bool is_moved(vector<int>& start_point, vector<int>& end_point);

// calculate the angle between two points.
int calc_angle(vector<int>& start_point, vector<int>& end_point);

// display progress bar of loop
void display_progress_bar(int bar_width, float progress);
