#ifndef CODE2IMG_H
#define CODE2IMG_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include "zcy_file.h"

using namespace std;
using namespace cv;

int min_line_len = 3;
int min_line_num = 3;
int img_width = 70;
int resize_width_ratio = 5;
int resize_height_ratio = 5;
string illegal_str[] = {"#",  "import", "namespace", "include"};

void trim(string &s)
{
    int index = 0;
    if( !s.empty())
    {
        while( (index = s.find(' ',index)) != string::npos)
        {
            s.erase(index,1);
        }
    }
}

bool is_legal_line(string line_str){
    bool ret = true;
    if(line_str.length()<min_line_len) ret = false;
    const char *show;
    for (string str : illegal_str){
        show = strstr(line_str.c_str(),str.c_str());
        if(show != NULL){
            ret = false;
        }
    }
    //    if(!ret) cout<<line_str<<endl;
    return ret;
}

int code2img(string filepath, Mat& ResImg, string save_dirpath = ""){
    ifstream ifs(filepath);
    string line_str;
    string::size_type iPos = filepath.find_last_of('\\') + 1;
    if(!iPos) iPos = filepath.find_last_of('/') + 1;
    string filename = filepath.substr(iPos, filepath.length() - iPos);
    string dirpath = filepath.substr(0, iPos);
    if(save_dirpath != "") dirpath = save_dirpath;
    if(!match("*/", dirpath.c_str())) dirpath = dirpath + "/";//linux
    string img_save_path = dirpath + filename + ".png";
    vector<string> str_vect;
    if(!ifs){
        cout<<"file doesn't exit"<<endl;
        return 0;
    }
    while(getline(ifs,line_str))   //按行读取,遇到换行符结束
    {
        trim(line_str);
        if(is_legal_line(line_str)) str_vect.push_back(line_str);
    }
    if(str_vect.size() < min_line_num){
        cout<<"file is too small, skip it!"<<filepath<<endl;
        return 0;
    }
    Mat ascii_mat = Mat::zeros(str_vect.size(), img_width, CV_8UC1);
    for (int row = 0; row < ascii_mat.rows; row++)
    {
        int col = 0;
        for (int str_vect_col = 0; str_vect_col < str_vect[row].length(); str_vect_col++){
            if(str_vect[row][str_vect_col]>32 && str_vect[row][str_vect_col]<127)
                if(col<img_width)
                    ascii_mat.at<uchar>(row, col++) = str_vect[row][str_vect_col];
        }
    }
    Size ResImgSiz = Size(ascii_mat.cols*resize_width_ratio, ascii_mat.rows*resize_height_ratio);
    ResImg = Mat::zeros(ResImgSiz, ascii_mat.type());
    resize(ascii_mat, ResImg, ResImgSiz, CV_INTER_CUBIC);
    normalize(ResImg,ResImg,255,0,NORM_MINMAX);
//    imshow("ResImg",ResImg);
//    waitKey(0);

    imwrite(img_save_path, ResImg);
    cout<<"img has been saved as "<<img_save_path<<endl;
    return 1;
}

#endif // CODE2IMG_H
