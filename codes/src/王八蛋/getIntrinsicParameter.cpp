#include "getIntrinsicParameter.h"

void getIntrinsicParameter(double focal_length, cv::Point2f principal_point, cv::Mat & K)
{
    K = cv::Mat_<double>(3,3);
    K.at<double>(0,0) = focal_length;
    K.at<double>(0,1) = 0;
    K.at<double>(0,2) = principal_point.x;
    K.at<double>(1,0) = 0;
    K.at<double>(1,1) = focal_length;
    K.at<double>(2,0) = 0;
    K.at<double>(2,1) = 0;
    K.at<double>(2,2) = 1;
    K.at<double>(1,2) = principal_point.y;

}
