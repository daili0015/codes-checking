#ifndef ZCY_COMPARE_H
#define ZCY_COMPARE_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include "zcy_sift.h"
#include "zcy_file.h"
#include "code2img.h"
#include "sift_stream.h"
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;
using namespace zcy_sift_lib;

const int search_range_index = 20;//判断valid_match时前后搜索比较多少个相邻mactch
const double search_range_row = 20*resize_height_ratio;//判断valid_match时前后搜索代码中的多少行
const double max_angle_diff = 0.05;//判断valid_match时最大允许斜率相差多少，相差过大说明match对应的不是同样的代码段
const int min_parallel_match = 4;//周围有多少个平行的match时判定为合法
const int suspicious_num = 10;//超过这一阈值，说明可能存在抄袭

struct match_pair{
    int point_y; //y轴坐标
    float angle; //线段斜率
};

bool is_valid_match(int index, vector<match_pair> all_match_pair){
    int parallel_match_count = 0;
    for(int i=index - search_range_index;i<index + search_range_index;i++){
        if(i>-1 && i<all_match_pair.size() && i!=index){
            if(all_match_pair[i].point_y-all_match_pair[index].point_y > search_range_row)
                continue;
            else{
//                cout<<fabs(all_match_pair[i].angle-all_match_pair[index].angle)<<" ";
                if(fabs(all_match_pair[i].angle-all_match_pair[index].angle)<max_angle_diff)
                    parallel_match_count++;
            }
        }
    }
    if(parallel_match_count>min_parallel_match)
        return true;
    else
        return false;
}

int sort_sift_point(vector<KeyPoint>& inlinerFirstKeypoint, vector<KeyPoint>& inlinerSecondKeypoint,
                    vector<DMatch>& inlinerMatches){
    KeyPoint tmp;
    DMatch tmp_match;
    for(int i=0;i<inlinerFirstKeypoint.size();i++){
        int miniPost = i;
        for(int j=i+1;j<inlinerFirstKeypoint.size();j++){
            if(inlinerFirstKeypoint[miniPost].pt.y>inlinerFirstKeypoint[j].pt.y)
                miniPost = j;
        }
        if(miniPost != i){
            tmp = inlinerFirstKeypoint[miniPost];
            inlinerFirstKeypoint[miniPost] = inlinerFirstKeypoint[i];
            inlinerFirstKeypoint[i] = tmp;
            //inlinerSecondKeypoint
            tmp = inlinerSecondKeypoint[miniPost];
            inlinerSecondKeypoint[miniPost] = inlinerSecondKeypoint[i];
            inlinerSecondKeypoint[i] = tmp;
            //inlinerMatches
            tmp_match = inlinerMatches[miniPost];
            inlinerMatches[miniPost] = inlinerMatches[i];
            inlinerMatches[i] = tmp_match;
        }
    }
}

double get_wall_time()
{
    struct timeval time ;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

string get_imgpath(string dat_file){
    string::size_type iPos = dat_file.find_last_of('.');
    if(!iPos) iPos = dat_file.find_last_of('.');
    return dat_file.substr(0, iPos)+".png";
}

string get_match_imgpath(string dat_file1, string dat_file2){
    string::size_type iPos = dat_file1.find_last_of('.');
    string part1 = dat_file1.substr(0, iPos);

    string::size_type iPos2 = dat_file2.find_last_of('.');
    string::size_type iPos3 = dat_file2.find_last_of('/')+1;
    string part2 = dat_file2.substr(iPos3, iPos2-iPos3);
    return part1+"-"+part2;
}

int zcy_com_with_dir(string file1, string dirpath, vector<string> compare_Dirs){
    vector<char*> legal_file_type = {"*.m",  "*.c", "*.cpp", "*.py", "*.h", "*.hpp"};
    sift_stream my_stream = sift_stream();
    for(int k=0;k<compare_Dirs.size();k++){
        vector<string> files, filenames;
        getAllFiles(dirpath+"/"+compare_Dirs[k]+"/", files, filenames);
        for(int i=0;i<files.size();i++){
            if(is_same_file_type(file1, filenames[i], legal_file_type)){
                //                cout<<file1<<files[i]<<endl;
                vector<KeyPoint> sift_Keypoint1,sift_Keypoint2;
                Mat firstDescriptor, secondDescriptor;
                my_stream.load_sift(file1, sift_Keypoint1, firstDescriptor);
                my_stream.load_sift(files[i], sift_Keypoint2, secondDescriptor);
                string img_path1 = get_imgpath(file1);
                string img_path2 = get_imgpath(files[i]);
                string match_img_path = get_match_imgpath(file1, files[i]);
                Mat firstImage = imread(img_path1);
                Mat secondImage = imread(img_path2);

                //match
                Ptr<DescriptorMatcher > matcher = DescriptorMatcher::create("BruteForce");
                Mat masks;
                Mat matcheImage;
                vector<DMatch> matches;
                matcher->match(firstDescriptor, secondDescriptor, matches, masks);

                ////////////////////////////////////////////////////////////////////////////////
                //第二步:RANSAC方法剔除outliner
                ////////////////////////////////////////////////////////////////////////////////
                //将vector转化成Mat
                Mat firstKeypointMat(matches.size(), 2, CV_32F), secondKeypointMat(matches.size(), 2, CV_32F);
                for (int i = 0; i<matches.size(); i++)
                {
                    firstKeypointMat.at<float>(i, 0) = sift_Keypoint1[matches[i].queryIdx].pt.x;
                    firstKeypointMat.at<float>(i, 1) = sift_Keypoint1[matches[i].queryIdx].pt.y;
                    secondKeypointMat.at<float>(i, 0) = sift_Keypoint2[matches[i].trainIdx].pt.x;
                    secondKeypointMat.at<float>(i, 1) = sift_Keypoint2[matches[i].trainIdx].pt.y;
                }
                //Calculate the fundamental Mat;
                vector<uchar> ransacStatus;
                Mat fundamentalMat = findFundamentalMat(firstKeypointMat, secondKeypointMat, ransacStatus, FM_RANSAC);
                //Calculate the number of outliner points;
                int outlinerCount = 0;
                if(ransacStatus.size()<3) return 1;
                for (int i = 0; i<matches.size(); i++)
                {
                    if (ransacStatus[i] == 0)
                    {
                        outlinerCount++;
                    }
                }
                //Calculate inliner points;
                vector<Point2f> firstInliner;
                vector<Point2f> secondInliner;
                vector<DMatch> inlinerMatches;
                int inlinerCount = matches.size() - outlinerCount;
                firstInliner.resize(inlinerCount);
                secondInliner.resize(inlinerCount);
                inlinerMatches.resize(inlinerCount);
                int index = 0;
                for (int i = 0; i<matches.size(); i++)
                {
                    if (ransacStatus[i] != 0)
                    {
                        firstInliner[index].x = firstKeypointMat.at<float>(i, 0);
                        firstInliner[index].y = firstKeypointMat.at<float>(i, 1);
                        secondInliner[index].x = secondKeypointMat.at<float>(i, 0);
                        secondInliner[index].y = secondKeypointMat.at<float>(i, 1);
                        inlinerMatches[index].queryIdx = index;
                        inlinerMatches[index].trainIdx = index;
                        index++;
                    }
                }

                vector<KeyPoint> inlinerFirstKeypoint(inlinerCount);
                vector<KeyPoint> inlinerSecondKeypoint(inlinerCount);
                KeyPoint::convert(firstInliner, inlinerFirstKeypoint);
                KeyPoint::convert(secondInliner, inlinerSecondKeypoint);
//                cout<<matches.size()<<" "<<inlinerMatches.size()<<endl;
//                drawMatches(firstImage, inlinerFirstKeypoint, secondImage, inlinerSecondKeypoint, inlinerMatches, matcheImage);
                ////////////////////////////////////////////////////////////////////////////////
                //第三步：平行线筛选器过滤掉错误匹配对
                ////////////////////////////////////////////////////////////////////////////////
                // sort_sift_point by inlinerFirstKeypoint.pt.y
                sort_sift_point(inlinerFirstKeypoint, inlinerSecondKeypoint, inlinerMatches);
                vector<match_pair> all_match_pair;
                vector<KeyPoint> valid_FirstKeypoint;
                vector<KeyPoint> valid_SecondKeypoint;
                vector<DMatch> valid_Matches;
                //init all_match_pair
                for(int i=0;i<inlinerCount;i++){
                    match_pair cur_match_pair;
                    cur_match_pair.point_y = inlinerFirstKeypoint[i].pt.y;
                    cur_match_pair.angle = (inlinerFirstKeypoint[i].pt.y-inlinerSecondKeypoint[i].pt.y)
                            /(inlinerFirstKeypoint[i].pt.x-inlinerSecondKeypoint[i].pt.x+firstImage.cols);
                    all_match_pair.push_back(cur_match_pair);
                }
                //get valid_match_pair
                int cur_index = 0;
                for(int i=0;i<inlinerCount;i++){
                    if(is_valid_match(i, all_match_pair)){
                        valid_FirstKeypoint.push_back(inlinerFirstKeypoint[i]);
                        valid_SecondKeypoint.push_back(inlinerSecondKeypoint[i]);
                        valid_Matches.push_back(inlinerMatches[i]);
                        valid_Matches[cur_index].queryIdx = cur_index;
                        valid_Matches[cur_index].trainIdx = cur_index;
                        cur_index++;
                    }
                }
                if(valid_Matches.size()>suspicious_num){
                    drawMatches(firstImage, valid_FirstKeypoint, secondImage, valid_SecondKeypoint, valid_Matches, matcheImage);
                    imwrite(match_img_path+"-"+to_string(valid_Matches.size())
                            +"-"+to_string(inlinerMatches.size())+"-"
                            +to_string(matches.size())+".png", matcheImage);
                    cout << "Suspicious result has been saved as: " << match_img_path
                            +"-"+to_string(valid_Matches.size())
                            +"-"+to_string(inlinerMatches.size())+"-"
                            +to_string(matches.size())+".png"<< endl;
                }
            }
        }
    }
    return 1;
}

int zcy_compare(string imgpath1, string imgpath2){
    //    double start, stop;
    //    start = get_wall_time();
    Mat matcheImage;
    string::size_type iPos1 = imgpath1.find_last_of('\\') + 1;
    if(!iPos1) iPos1 = imgpath1.find_last_of('/') + 1;
    string::size_type iPos1_point = imgpath1.find_first_of('.');
    string filename1 = imgpath1.substr(iPos1, iPos1_point-iPos1);
    string::size_type iPos2 = imgpath2.find_last_of('\\') + 1;
    if(!iPos2) iPos2 = imgpath2.find_last_of('/') + 1;
    string::size_type iPos2_point = imgpath2.find_first_of('.');
    string filename2 = imgpath2.substr(iPos2, iPos2_point-iPos2);
    string dir_path = imgpath1.substr(0, iPos1);

    Mat firstImage = imread(imgpath1);
    Mat secondImage = imread(imgpath2);
    if (firstImage.empty() || secondImage.empty())
    {
        cout << "no such img !!!" << endl;
        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //第一步:获取SIFT特征
    ////////////////////////////////////////////////////////////////////////////////
    //difine a sift detector
    zcy_sift my_sift = zcy_sift();

    vector<KeyPoint> firstKeypoint, secondKeypoint;
    my_sift.detect(firstImage, firstKeypoint);
    my_sift.detect(secondImage, secondKeypoint);

    Mat firstDescriptor, secondDescriptor;
    my_sift.compute(firstImage,firstKeypoint,firstDescriptor);
    my_sift.compute(secondImage,secondKeypoint,secondDescriptor);
    Ptr<DescriptorMatcher > matcher = DescriptorMatcher::create("BruteForce");
    Mat masks;
    vector<DMatch> matches;
    matcher->match(firstDescriptor, secondDescriptor, matches, masks);

    drawMatches(firstImage, firstKeypoint, secondImage, secondKeypoint, matches, matcheImage);
    imwrite(dir_path+to_string(matches.size())+"-"+filename1+"*"+filename2+"ransac.png", matcheImage);
    ////////////////////////////////////////////////////////////////////////////////
    //第二步:RANSAC方法剔除outliner
    ////////////////////////////////////////////////////////////////////////////////

    //将vector转化成Mat
    Mat firstKeypointMat(matches.size(), 2, CV_32F), secondKeypointMat(matches.size(), 2, CV_32F);
    for (int i = 0; i<matches.size(); i++)
    {
        firstKeypointMat.at<float>(i, 0) = firstKeypoint[matches[i].queryIdx].pt.x;
        firstKeypointMat.at<float>(i, 1) = firstKeypoint[matches[i].queryIdx].pt.y;
        secondKeypointMat.at<float>(i, 0) = secondKeypoint[matches[i].trainIdx].pt.x;
        secondKeypointMat.at<float>(i, 1) = secondKeypoint[matches[i].trainIdx].pt.y;
    }
    //Calculate the fundamental Mat;
    vector<uchar> ransacStatus;
    Mat fundamentalMat = findFundamentalMat(firstKeypointMat, secondKeypointMat, ransacStatus, FM_RANSAC);
    //    cout << fundamentalMat << endl;
    //Calculate the number of outliner points;
    int outlinerCount = 0;
    for (int i = 0; i<matches.size(); i++)
    {
        if (ransacStatus[i] == 0)
        {
            outlinerCount++;
        }
    }
    //Calculate inliner points;
    vector<Point2f> firstInliner;
    vector<Point2f> secondInliner;
    vector<DMatch> inlinerMatches;
    int inlinerCount = matches.size() - outlinerCount;
    firstInliner.resize(inlinerCount);
    secondInliner.resize(inlinerCount);
    inlinerMatches.resize(inlinerCount);
    int index = 0;
    for (int i = 0; i<matches.size(); i++)
    {
        if (ransacStatus[i] != 0)
        {
            firstInliner[index].x = firstKeypointMat.at<float>(i, 0);
            firstInliner[index].y = firstKeypointMat.at<float>(i, 1);
            secondInliner[index].x = secondKeypointMat.at<float>(i, 0);
            secondInliner[index].y = secondKeypointMat.at<float>(i, 1);
            inlinerMatches[index].queryIdx = index;
            inlinerMatches[index].trainIdx = index;
            index++;
        }
    }

    vector<KeyPoint> inlinerFirstKeypoint(inlinerCount);
    vector<KeyPoint> inlinerSecondKeypoint(inlinerCount);
    KeyPoint::convert(firstInliner, inlinerFirstKeypoint);
    KeyPoint::convert(secondInliner, inlinerSecondKeypoint);
    cout<<matches.size()<<" "<<inlinerMatches.size()<<endl;
    drawMatches(firstImage, inlinerFirstKeypoint, secondImage, inlinerSecondKeypoint, inlinerMatches, matcheImage);
    imwrite(dir_path+to_string(inlinerMatches.size())+"-"
            +to_string(matches.size())+"-"+filename1+"*"+filename2+"ransac.png", matcheImage);
    imshow("matches", matcheImage);
//    waitKey(0);
//    cout << "Now you can check result: " << dir_path+to_string(inlinerMatches.size())+"-"
//            +to_string(matches.size())+"-"+filename1+"*"+filename2+".png" << endl;
    ////////////////////////////////////////////////////////////////////////////////
    //第三步：平行线筛选器过滤掉错误匹配对
    ////////////////////////////////////////////////////////////////////////////////
    // sort_sift_point by inlinerFirstKeypoint.pt.y
    sort_sift_point(inlinerFirstKeypoint, inlinerSecondKeypoint, inlinerMatches);
    vector<match_pair> all_match_pair;
    vector<KeyPoint> valid_FirstKeypoint;
    vector<KeyPoint> valid_SecondKeypoint;
    vector<DMatch> valid_Matches;
    //init all_match_pair
    for(int i=0;i<inlinerCount;i++){
        match_pair cur_match_pair;
        cur_match_pair.point_y = inlinerFirstKeypoint[i].pt.y;
        cur_match_pair.angle = (inlinerFirstKeypoint[i].pt.y-inlinerSecondKeypoint[i].pt.y)
                /(inlinerFirstKeypoint[i].pt.x-inlinerSecondKeypoint[i].pt.x+firstImage.cols);
        all_match_pair.push_back(cur_match_pair);
    }
    //get valid_match_pair
    int cur_index = 0;
    for(int i=0;i<inlinerCount;i++){
        if(is_valid_match(i, all_match_pair)){
            valid_FirstKeypoint.push_back(inlinerFirstKeypoint[i]);
            valid_SecondKeypoint.push_back(inlinerSecondKeypoint[i]);
            valid_Matches.push_back(inlinerMatches[i]);
            valid_Matches[cur_index].queryIdx = cur_index;
            valid_Matches[cur_index].trainIdx = cur_index;
            cur_index++;
        }
    }
    cout<<matches.size()<<" "<<inlinerMatches.size()<<" "<<valid_Matches.size()<<endl;
    drawMatches(firstImage, valid_FirstKeypoint, secondImage, valid_SecondKeypoint, valid_Matches, matcheImage);
    imwrite(dir_path+to_string(valid_Matches.size())+"-"+to_string(inlinerMatches.size())+"-"
            +to_string(matches.size()) +"-"+filename1+"*"+filename2+".png", matcheImage);
    imshow("valid_matches", matcheImage);
    waitKey(0);
    return 1;
}

#endif // ZCY_COMPARE_H
