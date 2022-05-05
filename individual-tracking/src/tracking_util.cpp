#include <iostream>
#include <chrono>
#include <numeric>
#include <vector>
#include <deque>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using std::cout;
using std::endl;
using std::string;

bool is_not_digit(char c)
{
    return !isdigit(c);
}

bool numeric_string_compare(const string &s1, const string &s2)
{
    /** numerical compare two strings.
     * ex)
     *   s1 = "32.csv"  s2 = "1265.csv"
     *   s1 < s2
     **/

    if (s1.empty() || s2.empty())
    {
        cout << "ERROR: input string is empty." << endl;
        exit(1);
    }

    string::const_iterator it1 = s1.begin(), it2 = s2.begin();

    if (isdigit(s1.at(0)) && isdigit(s2.at(0)))
    {
        int n1, n2;
        std::stringstream ss(s1);
        ss >> n1;
        ss.clear();
        ss.str(s2);
        ss >> n2;

        if (n1 != n2)
            return n1 < n2;

        it1 = find_if(s1.begin(), s2.end(), is_not_digit);
        it2 = find_if(s2.begin(), s2.end(), is_not_digit);
    }

    return lexicographical_compare(it1, s1.end(), it2, s2.end());
}

void display_video_info(string file_path, int width, int height, int total_frame, int fourcc, double fps)
{
    cout << "\n*******************************************" << endl;
    cout << "VIDEO PATH : " << file_path << endl;
    cout << "WIDTH      : " << width << endl;
    cout << "HEIGHT     : " << height << endl;
    cout << "TOTAL FRAME: " << total_frame << endl;
    cout << "FOURCC     : " << fourcc << endl;
    cout << "FPS        : " << fps << endl;
    cout << "*******************************************\n"
         << endl;
}

void make_file_vec(string input_dircpath, std::vector<string> &file_vec, string file_extension)
{
    /**
     * make a vector of filepath with the specified extension.
     *
     * input:
     *  input_dircpath : the direcory path is "const"
     *  file_extension : ex) .csv
     **/

    // vector<string> file_vec;
    DIR *dp;
    dirent *entry;
    string entry_name;

    dp = opendir(input_dircpath.c_str());
    if (dp == NULL)
    {
        cout << "ERROR: can not open the file. please check directory path." << endl;
        cout << "input path: " << input_dircpath << endl;
        exit(1);
    }

    do
    {
        entry = readdir(dp);
        if (entry != NULL)
        {
            entry_name = entry->d_name;
            if (entry_name.find(file_extension) != string::npos)
            {
                file_vec.push_back(entry_name);
            }
        }
    } while (entry != NULL);
    closedir(dp);

    // ascending numarical sort
    sort(file_vec.begin(), file_vec.end(), numeric_string_compare);

    // NEED FIX
    for (int i = 0; i < file_vec.size(); i++)
    {
        file_vec.at(i).insert(0, input_dircpath);
    }
}

void read_csv(string input_csv_file_path, std::vector<std::vector<int>> &table, const char delimiter = ',')
{
    /**
     * to make vector table of csv data
     *
     * input:
     *      input_csv_file_path :
     *      table               :
     *      delimeter           :
     **/
    std::fstream filestream(input_csv_file_path);
    if (!filestream.is_open())
    {
        cout << "ERROR: can not open file (input csv). please check file path." << endl;
        cout << "input path: " << input_csv_file_path << endl;
        exit(1);
    }

    while (!filestream.eof())
    {
        std::string buffer;
        filestream >> buffer;

        std::vector<int> record;
        std::istringstream streambuffer(buffer);
        string token;
        while (getline(streambuffer, token, delimiter))
        {
            record.push_back(atoi(token.c_str()));
        }
        if (!record.empty())
            table.push_back(record);
    }
}

void write_csv(std::vector<float> &data_vec, std::vector<int> &frame_num_vec,
               std::vector<string> &header_vec, string output_csv_path)
{
    /**
     * write csv file by vector data
     *
     * input:
     *   data_vec             : vector of target value (mean, var, max)
     *   frame_num_vec        : record from which frame the value of vector was generated
     *   output_csv_file_path : absolute path
     **/

    // check vector size
    assert(data_vec.size() <= frame_num_vec.size());

    std::ofstream ofs(output_csv_path);
    if (ofs)
    {
        // insert header
        ofs << header_vec.at(0) << "," << header_vec.at(1) << endl;
        for (unsigned int i = 0; i < data_vec.size(); ++i)
        {
            ofs << frame_num_vec.at(i) << "," << data_vec.at(i) << endl;
        }
    }
    else
    {
        cout << "ERROR: can not open file (output csv). please check file path." << endl;
        cout << "input path: " << output_csv_path << endl;
        exit(1);
    }

    ofs.close();
    cout << "DONE: " << output_csv_path << endl;
}

void write_csv_2d(std::vector<std::vector<int>> &data_vec, std::vector<int> &frame_num_vec,
                  std::vector<string> &header_vec, string output_csv_path)
{
    /**
     * write csv file by 2D vector data
     *
     * input:
     *   data_vec             : 2D vector of target value (degree)
     *   frame_num_vec        : record from which frame the value of vector was generated
     *   output_csv_file_path : absolute path
     **/

    // check vector size
    assert(data_vec.size() <= frame_num_vec.size());

    std::ofstream ofs(output_csv_path);
    if (ofs)
    {
        // insert header
        ofs << header_vec.at(0) << "," << header_vec.at(1) << endl;
        for (unsigned int i = 0; i < data_vec.size(); ++i)
        {
            ofs << frame_num_vec.at(i) << ",";
            for (unsigned int j = 0; j < data_vec.at(i).size(); ++j)
            {
                ofs << (int)data_vec.at(i).at(j) << ",";
            }
            ofs << endl;
        }
    }
    else
    {
        cout << "ERROR: can not open file (output csv). please check file path." << endl;
        cout << "input path: " << output_csv_path << endl;
        exit(1);
    }

    ofs.close();
    cout << "DONE: " << output_csv_path << endl;
}

std::vector<float> point_norm(std::vector<std::vector<int>> &point_vec)
{
    /** caluculate norm between two point
     *
     * input:
     *   point_vec:
     **/

    float pow_norm = 0.0;
    std::vector<float> norm_vec;
    for (int i = 0; i < point_vec.size(); ++i)
    {
        pow_norm = std::pow(float(point_vec.at(i).at(0)), 2.0) + std::pow(float(point_vec.at(i).at(1)), 2.0);
        norm_vec.push_back(std::sqrt(pow_norm));
    }

    assert(point_vec.size() == norm_vec.size());

    return norm_vec;
}

float vec_variance(std::vector<float> &value_vec, float mean)
{
    /** caluculate variance of value vec
     *
     * input:
     *   value_vec:
     *   mean     : mean of value_vec
     **/

    float var = 0.0;
    for (int i = 0; i < value_vec.size(); ++i)
    {
        var += std::pow((value_vec.at(i) - mean), 2);
    }

    return var / value_vec.size();
}

bool is_moved(std::vector<int> &start_point, std::vector<int> &end_point)
{
    /** traget has moved or not.
     * input:
     *  start_point: cordinate of prev frame.
     *  end_point  : cordinate of current frame.
     **/
    if ((start_point.at(0) == end_point.at(0)) && (start_point.at(1) == end_point.at(1)))
    {
        // NOT moved
        return false;
    }
    else
    {
        // moved
        return true;
    }
}

int calc_degree(std::vector<int> &start_point, std::vector<int> &end_point)
{
    /** caluculate degree by two point
     * input:
     *  start_point:
     *  end_point  :
     *
     * argument of atan2 is (y, x)
     **/
    float radian_angle = std::atan2(end_point.at(1) - start_point.at(1), end_point.at(0) - start_point.at(0));
    float degree = (radian_angle * 180) / M_PI;
    if (degree < 0)
    {
        degree += 360;
    }
    return (int)degree;
}