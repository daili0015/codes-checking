#ifndef SIFT_STREAM_H
#define SIFT_STREAM_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<string> split(string str, string pattern)
{
    vector<string> ret;
    if(pattern.empty()) return ret;
    size_t start=0,index=str.find_first_of(pattern,0);
    while(index!=str.npos)
    {
        if(start!=index)
            ret.push_back(str.substr(start,index-start));
        start=index+1;
        index=str.find_first_of(pattern,start);
    }
    if(!str.substr(start).empty())
        ret.push_back(str.substr(start));
    return ret;
}

class sift_stream
{
public:
    sift_stream() {}
    int save_sift(string save_path, vector<KeyPoint> sift_Keypoint, Mat Descriptor){
        ofstream ply(save_path);
        ply.precision(6);
        for(int i=0;i<sift_Keypoint.size();i++){
            ply << sift_Keypoint[i].angle << " ";
            ply << sift_Keypoint[i].octave << " ";
            ply << sift_Keypoint[i].size << " ";
            ply << sift_Keypoint[i].response << " ";
            ply << sift_Keypoint[i].pt.x << " ";
            ply << sift_Keypoint[i].pt.y << " ";
            ply << endl;
        }
        ply << "Descriptor" << std::endl;
        ply << Descriptor;
        ply.close();
        return 1;
    }
    int load_sift(string load_path, vector<KeyPoint>& sift_Keypoint, Mat& Descriptor){
        ifstream ply(load_path);
        ply.precision(6);
        std::string strLine;
        std::getline(ply,strLine);
        int point_count = 0;
        while(strLine.compare("Descriptor")!=0){
            point_count++;
            KeyPoint cur_keypoint;
            std::vector<std::string> dataline_vect = split(strLine, " ");
            cur_keypoint.angle=std::atof(dataline_vect[0].c_str());
            cur_keypoint.octave=std::atof(dataline_vect[1].c_str());
            cur_keypoint.size=std::atof(dataline_vect[2].c_str());
            cur_keypoint.response=std::atof(dataline_vect[3].c_str());
            cur_keypoint.pt.x=std::atof(dataline_vect[4].c_str());
            cur_keypoint.pt.y=std::atof(dataline_vect[5].c_str());
            sift_Keypoint.push_back(cur_keypoint);
            std::getline(ply,strLine);
        }
        Mat load_Descriptor = Mat::zeros(point_count, 128, CV_8UC1);
        for(int i=0;i<point_count;i++){
            std::getline(ply,strLine);
            std::vector<std::string> dataline_vect = split(strLine, " ");
            for(int k=0;k<128;k++){
                load_Descriptor.at<uchar>(i, k)= std::atof(dataline_vect[k].c_str());
            }
        }
        Descriptor = load_Descriptor.clone();
        ply.close();

        return 1;
    }
};


#endif // SIFT_STREAM_H
