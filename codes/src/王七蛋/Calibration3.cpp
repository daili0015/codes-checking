#ifndef _CALIBRATE_H
#define _CALIBRATE_H
#include <opencv2/opencv.hpp>


cv::Point2f getLineIntersection(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
{
    cv::Point2f vp;
    float x0, x1, x2, x3, y0, y1, y2, y3;
    x0 = p0.x;   y0 = p0.y;
    x1 = p1.x;   y1 = p1.y;
    x2 = p2.x;   y2 = p2.y;
    x3 = p3.x;   y3 = p3.y;
    vp.y = ( (y0-y1)*(y3-y2)*x0 + (y3-y2)*(x1-x0)*y0 + (y1-y0)*(y3-y2)*x2 + (x2-x3)*(y1-y0)*y2 ) / ( (x1-x0)*(y3-y2) + (y0-y1)*(x3-x2) );
    vp.x = x2 + (x3-x2)*(vp.y-y2) / (y3-y2);
    return vp;
}

double getFocalLength(cv::Point2f vx, cv::Point2f vy)
{
    cv::Point2f v1 = vx;
    cv::Point2f v2 = vy;

    return sqrt( -1*(v1.x * v2.x + v1.y * v2.y) );
}

void adjustRotateMat(cv::Mat srcR, cv::Mat &dstR)
{
    cv::Mat x_axis = cv::Mat_<double>(3,1);
    cv::Mat y_axis = cv::Mat_<double>(3,1);
    cv::Mat z_axis = cv::Mat_<double>(3,1);
    x_axis.at<double>(0,0) = srcR.at<double>(0,0);
    x_axis.at<double>(1,0) = srcR.at<double>(1,0);
    x_axis.at<double>(2,0) = srcR.at<double>(2,0);

    y_axis.at<double>(0,0) = srcR.at<double>(0,1);
    y_axis.at<double>(1,0) = srcR.at<double>(1,1);
    y_axis.at<double>(2,0) = srcR.at<double>(2,1);

    z_axis.at<double>(0,0) = srcR.at<double>(0,2);
    z_axis.at<double>(1,0) = srcR.at<double>(1,2);
    z_axis.at<double>(2,0) = srcR.at<double>(2,2);

    int biggest_index=1;
    double biggest_val = fabs((double)x_axis.at<double>(1,0));
    if (fabs((double)y_axis.at<double>(1, 0)) > biggest_val)
    {
        biggest_index = 2;
        biggest_val = fabs((double)y_axis.at<double>(1, 0));
    }
    if (fabs((double)z_axis.at<double>(1, 0)) > biggest_val)
    {
        biggest_index = 3;
        biggest_val = fabs((double)z_axis.at<double>(1, 0));
    }
    int symble;
    cv::Mat v1 = cv::Mat_<double>(3,1);
    cv::Mat v2 = cv::Mat_<double>(3,1);
    cv::Mat v3 = cv::Mat_<double>(3,1);
    switch(biggest_index)
    {
    case 1:
        if(x_axis.at<double>(1,0) > 0) /**竖直向下*/
            symble = -1;
        else
        {
            symble = 1;
        }
        z_axis.at<double>(0,0) = symble * x_axis.at<double>(0,0);
        z_axis.at<double>(1,0) = symble * x_axis.at<double>(1,0);
        z_axis.at<double>(2,0) = symble * x_axis.at<double>(2,0);
        v1.at<double>(0,0) = y_axis.at<double>(0,0);
        v1.at<double>(1,0) = y_axis.at<double>(1,0);
        v1.at<double>(2,0) = y_axis.at<double>(2,0);

        v2.at<double>(0,0) = z_axis.at<double>(0,0);
        v2.at<double>(1,0) = z_axis.at<double>(1,0);
        v2.at<double>(2,0) = z_axis.at<double>(2,0);
        /**cross pruduct v1 * v2 */
        v3 = v1.cross(v2);
        //v3 = v2.cross(v1);
        x_axis = v3;
        break;
    case 2:
        if(y_axis.at<double>(1,0) > 0)
        {
            symble = -1;
        }
        else
        {
            symble = 1;
        }
        z_axis.at<double>(0,0) = symble * y_axis.at<double>(0,0);
        z_axis.at<double>(1,0) = symble * y_axis.at<double>(1,0);
        z_axis.at<double>(1,0) = symble * y_axis.at<double>(2,0);
        v1.at<double>(0,0) = z_axis.at<double>(0,0);
        v1.at<double>(1,0) = z_axis.at<double>(1,0);
        v1.at<double>(2,0) = z_axis.at<double>(2,0);

        v2.at<double>(0,0) = x_axis.at<double>(0,0);
        v2.at<double>(1,0) = x_axis.at<double>(1,0);
        v2.at<double>(2,0) = x_axis.at<double>(2,0);

        v3 = v1.cross(v2);
        y_axis = v3.clone();
        break;
    case 3:
        if(z_axis.at<double>(1,0) > 0)
        {
            z_axis.at<double>(0,0) = -1 * z_axis.at<double>(0,0);
            z_axis.at<double>(1,0) = -1 * z_axis.at<double>(1,0);
            z_axis.at<double>(2,0) = -1 * z_axis.at<double>(2,0);
            v1.at<double>(0,0) = z_axis.at<double>(0,0);
            v1.at<double>(1,0) = z_axis.at<double>(1,0);
            v1.at<double>(2,0) = z_axis.at<double>(2,0);
            v2.at<double>(0,0) = x_axis.at<double>(0,0);
            v2.at<double>(1,0) = x_axis.at<double>(1,0);
            v2.at<double>(2,0) = x_axis.at<double>(2,0);
            v3 = v1.cross(v2);
            y_axis = v3.clone();

        }

        break;
    default:
        std::cout<<">> case default"<<std::endl;
        break;

    }

    dstR.at<double>(0,0) = x_axis.at<double>(0,0);
    dstR.at<double>(1,0) = x_axis.at<double>(1,0);
    dstR.at<double>(2,0) = x_axis.at<double>(2,0);

    dstR.at<double>(0,1) = y_axis.at<double>(0,0);
    dstR.at<double>(1,1) = y_axis.at<double>(1,0);
    dstR.at<double>(2,1) = y_axis.at<double>(2,0);

    dstR.at<double>(0,2) = z_axis.at<double>(0,0);
    dstR.at<double>(1,2) = z_axis.at<double>(1,0);
    dstR.at<double>(2,2) = z_axis.at<double>(2,0);
}

void getRotateMat(cv::Point2f vx, cv::Point2f vy, double focal_length, cv::Mat &R)
{
    cv::Mat r1 = cv::Mat_<double>(3,1);
    cv::Mat r2 = cv::Mat_<double>(3,1);
    cv::Mat r3 = cv::Mat_<double>(3,1);

    r1.at<double>(0,0) = vx.x;
    r1.at<double>(1,0) = vx.y;
    r1.at<double>(2,0) = focal_length;
    cv::normalize(r1,r1);

    r2.at<double>(0,0) = vy.x;
    r2.at<double>(1,0) = vy.y;
    r2.at<double>(2,0) = focal_length;
    cv::normalize(r2,r2);

    r3 = r1.cross(r2);
    R = cv::Mat_<double>(3,3);
    R.at<double>(0,0) = r1.at<double>(0,0);
    R.at<double>(1,0) = r1.at<double>(1,0);
    R.at<double>(2,0) = r1.at<double>(2,0);

    R.at<double>(0,1) = r2.at<double>(0,0);
    R.at<double>(1,1) = r2.at<double>(1,0);
    R.at<double>(2,1) = r2.at<double>(2,0);

    R.at<double>(0,2) = r3.at<double>(0,0);
    R.at<double>(1,2) = r3.at<double>(1,0);
    R.at<double>(2,2) = r3.at<double>(2,0);
    adjustRotateMat(R,R);
}


bool getTranslationVector(const cv::Point2f origin, const cv::Point2f v1, const cv::Mat R, const double focal_length, cv::Mat &T)
{

    if (fabs((double)(origin.x - v1.x))<20 && fabs((double)(origin.y - v1.y))<20)
        return false;

    /**x-axis is a line from origin to v1 in image space*/
    cv::Mat line_x_axis = cv::Mat_<double>(4,1);
    line_x_axis.at<double>(0,0) = origin.x;
    line_x_axis.at<double>(1,0) = origin.y;
    line_x_axis.at<double>(2,0) = v1.x;
    line_x_axis.at<double>(3,0) = v1.y;

    /**direction of x-axis */
    cv::Mat dir_x_axis = cv::Mat_<double>(2,1);
    dir_x_axis.at<double>(0,0) = v1.x - origin.x;
    dir_x_axis.at<double>(1,0) = v1.y - origin.y;
    cv::Mat length = cv::Mat_<double>(1,1);
    length = dir_x_axis.t()*dir_x_axis;
    dir_x_axis.at<double>(0,0) /= sqrt(length.at<double>(0,0));
    dir_x_axis.at<double>(1,0) /= sqrt(length.at<double>(0,0));

    /**p为x轴上一点在图像上的投影坐标,其在世界坐标系内的坐标为（scale*d,0,0）**/
    cv::Point2f p;
    /**p点同原点在图像上的距离**/
    int d=10;
    int scale = d * 30;

    p.x = (int)(origin.x + d*dir_x_axis.at<double>(0,0) + 0.5);
    p.y = (int)(origin.y + d*dir_x_axis.at<double>(1,0) + 0.5);

    /**translation vector*/
    if (fabs((double)(p.x - origin.x)) > fabs((double)(p.y - origin.y)))
    {
        T.at<double>(2,0) = scale  * (R.at<double>(0,0)*focal_length - R.at<double>(2,0)*p.x)/(double)(p.x - origin.x);
    }
    else
    {
        T.at<double>(2,0) = scale  * (R.at<double>(1,0)*focal_length - R.at<double>(2,0)*p.y)/(double)(p.y - origin.y);
    }

    if(T.at<double>(2,0) < 0)
    {
        T.at<double>(2,0) =  -1 * T.at<double>(2,0);
    }
    if(T.at<double>(2,0) < focal_length)
    {
        return false;
    }
    T.at<double>(0,0) = origin.x * T.at<double>(2,0) / focal_length;
    T.at<double>(1,0) = origin.y * T.at<double>(2,0) / focal_length;
    return true;
}

void getIntrinsicParameter(double focal_length, cv::Point2f principal_point, cv::Mat & K)
{
    K = cv::Mat_<double>(3,3);
    K.at<double>(0,0) = focal_length;
    K.at<double>(0,1) = 0;
    K.at<double>(0,2) = principal_point.x;
    K.at<double>(1,0) = 0;
    K.at<double>(1,1) = focal_length;
    K.at<double>(1,2) = principal_point.y;
    K.at<double>(2,0) = 0;
    K.at<double>(2,1) = 0;
    K.at<double>(2,2) = 1;
}

cv::Point2f getVanishPoint(cv::Point2f v0, cv::Point2f v1, cv::Point2f v2, cv::Point2f v3)
{
    cv::Point2f vp;
    vp = getLineIntersection(v0, v1, v2, v3);
    return vp;
}

void zcy_calibrate(cv::Mat frame, cv::Point2f data_point[9], double &focal_length, cv::Mat &R, cv::Mat &T, cv::Mat &K)
{
    assert(R.rows == 3 && R.cols == 3);
    assert(T.rows == 3 && T.cols == 1);
    cv::Point2f point[9];
    cv::Point2f principal_point;
    principal_point.x = (double)frame.cols/2.0;
    principal_point.y = (double)frame.rows/2.0;
    std::cout<<">> p = "<<principal_point<<std::endl;
    for(int i=0; i<11; i++)
    {
        point[i] = data_point[i]-principal_point;
    }
    cv::Point2f origin = point[8];

    cv::Point2f v1 = getVanishPoint(point[0], point[1], point[2], point[3]);
    cv::Point2f v2 = getVanishPoint(point[4], point[5], point[6], point[7]);

    focal_length = getFocalLength(v1, v2);
    getRotateMat(v1, v2, focal_length, R);
    getTranslationVector(origin,v1, R, focal_length, T);
    getIntrinsicParameter(focal_length, principal_point, K);
}

#endif // _CALIBRATE_H
