#pragma once
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#define SCALE 100
using namespace cv;
using namespace std;

//标定程序，输入图片，数据点求出焦距和相机内外参K R T矩阵
void Calibrate(Mat frame, Point2f data_point[11], double &f_len, Mat &K, Mat &R, Mat &T);
//输入两条直线上的四个点，求出灭点
Point2f getVanishPoint(Point2f v0, Point2f v1, Point2f v2, Point2f v3);
//根据已知的焦距和坐标原点求解矩阵内参
void getIntrinsicParameter(double f_len, Point2f principal_point, Mat & K);
//由焦距和两个灭点，求出相机外参旋转矩阵
void getRotateMat(Point2f vx, Point2f vy, double f_len, Mat &R);
//根据原点、灭点和旋转矩阵和焦距求解相机外参平移向量T
bool getTranslationVector(const Point2f origin, const Point2f v1, const Mat R, const double f_len, Mat &T);
//旋转矩阵调整
void adjustRotateMat(Mat srcR, Mat &dstR);
//根据灭点求焦距
double getF_Len(Point2f vx, Point2f vy);

bool getTranslationVector(const Point2f origin, const Point2f v1, const Mat R, const double f_len, Mat &T)
{
	if (fabs((double)(origin.x - v1.x))<20 && fabs((double)(origin.y - v1.y))<20)
		return false;

	/**x-axis is a line from origin to v1 in image space*/
	Mat line_x_axis = Mat_<double>(4, 1);
	line_x_axis.at<double>(0, 0) = origin.x;
	line_x_axis.at<double>(1, 0) = origin.y;
	line_x_axis.at<double>(2, 0) = v1.x;
	line_x_axis.at<double>(3, 0) = v1.y;

	/**direction of x-axis */
	Mat dir_x_axis = Mat_<double>(2, 1);
	dir_x_axis.at<double>(0, 0) = v1.x - origin.x;
	dir_x_axis.at<double>(1, 0) = v1.y - origin.y;
	Mat length = Mat_<double>(1, 1);
	length = dir_x_axis.t()*dir_x_axis;
	dir_x_axis.at<double>(0, 0) /= sqrt(length.at<double>(0, 0));
	dir_x_axis.at<double>(1, 0) /= sqrt(length.at<double>(0, 0));

	/**p为x轴上一点在图像上的投影坐标,其在世界坐标系内的坐标为（scale*d,0,0）**/
	Point2f p;
	/**p点同原点在图像上的距离**/
	int d = 20;
	int scale = d * SCALE;

	p.x = (int)(origin.x + d*dir_x_axis.at<double>(0, 0) + 0.5);
	p.y = (int)(origin.y + d*dir_x_axis.at<double>(1, 0) + 0.5);

	/**translation vector*/
	if (fabs((double)(p.x - origin.x)) > fabs((double)(p.y - origin.y)))
	{
		T.at<double>(2, 0) = scale  * (R.at<double>(0, 0)*f_len - R.at<double>(2, 0)*p.x) / (double)(p.x - origin.x);
	}
	else
	{
		T.at<double>(2, 0) = scale  * (R.at<double>(1, 0)*f_len - R.at<double>(2, 0)*p.y) / (double)(p.y - origin.y);
	}

	if (T.at<double>(2, 0) < 0)
	{
		T.at<double>(2, 0) = -1 * T.at<double>(2, 0);
	}
	if (T.at<double>(2, 0) < f_len)
	{
		return false;
	}
	T.at<double>(0, 0) = origin.x * T.at<double>(2, 0) / f_len;
	T.at<double>(1, 0) = origin.y * T.at<double>(2, 0) / f_len;
	return true;
}

Point2f getVanishPoint(Point2f v0, Point2f v1, Point2f v2, Point2f v3)
{
	Point2f vp;
	vp.y = ((v0.y - v1.y)*(v3.y - v2.y)*v0.x + (v3.y - v2.y)*(v1.x - v0.x)*v0.y + (v1.y - v0.y)*(v3.y - v2.y)*v2.x + (v2.x - v3.x)*(v1.y - v0.y)*v2.y) / ((v1.x - v0.x)*(v3.y - v2.y) + (v0.y - v1.y)*(v3.x - v2.x));
	vp.x = v2.x + (v3.x - v2.x)*(vp.y - v2.y) / (v3.y - v2.y);
	return vp;
}

void adjustRotateMat(cv::Mat srcR, cv::Mat &dstR)
{
	Mat x_axis = Mat_<double>(3, 1);
	Mat y_axis = Mat_<double>(3, 1);
	Mat z_axis = Mat_<double>(3, 1);
	x_axis.at<double>(0, 0) = srcR.at<double>(0, 0);
	x_axis.at<double>(1, 0) = srcR.at<double>(1, 0);
	x_axis.at<double>(2, 0) = srcR.at<double>(2, 0);

	y_axis.at<double>(0, 0) = srcR.at<double>(0, 1);
	y_axis.at<double>(1, 0) = srcR.at<double>(1, 1);
	y_axis.at<double>(2, 0) = srcR.at<double>(2, 1);

	z_axis.at<double>(0, 0) = srcR.at<double>(0, 2);
	z_axis.at<double>(1, 0) = srcR.at<double>(1, 2);
	z_axis.at<double>(2, 0) = srcR.at<double>(2, 2);

	int biggest_index = 1;
	double biggest_val = fabs((double)x_axis.at<double>(1, 0));
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
	Mat v1 = Mat_<double>(3, 1);
	Mat v2 = Mat_<double>(3, 1);
	Mat v3 = Mat_<double>(3, 1);
	switch (biggest_index)
	{
	case 1:
		if (x_axis.at<double>(1, 0) > 0) /**竖直向下*/
			symble = -1;
		else
		{
			symble = 1;
		}
		z_axis.at<double>(0, 0) = symble * x_axis.at<double>(0, 0);
		z_axis.at<double>(1, 0) = symble * x_axis.at<double>(1, 0);
		z_axis.at<double>(2, 0) = symble * x_axis.at<double>(2, 0);
		v1.at<double>(0, 0) = y_axis.at<double>(0, 0);
		v1.at<double>(1, 0) = y_axis.at<double>(1, 0);
		v1.at<double>(2, 0) = y_axis.at<double>(2, 0);

		v2.at<double>(0, 0) = z_axis.at<double>(0, 0);
		v2.at<double>(1, 0) = z_axis.at<double>(1, 0);
		v2.at<double>(2, 0) = z_axis.at<double>(2, 0);
		/**cross pruduct v1 * v2 */
		v3 = v1.cross(v2);
		//v3 = v2.cross(v1);
		x_axis = v3;
		break;
	case 2:
		if (y_axis.at<double>(1, 0) > 0)
		{
			symble = -1;
		}
		else
		{
			symble = 1;
		}
		z_axis.at<double>(0, 0) = symble * y_axis.at<double>(0, 0);
		z_axis.at<double>(1, 0) = symble * y_axis.at<double>(1, 0);
		z_axis.at<double>(1, 0) = symble * y_axis.at<double>(2, 0);
		v1.at<double>(0, 0) = z_axis.at<double>(0, 0);
		v1.at<double>(1, 0) = z_axis.at<double>(1, 0);
		v1.at<double>(2, 0) = z_axis.at<double>(2, 0);

		v2.at<double>(0, 0) = x_axis.at<double>(0, 0);
		v2.at<double>(1, 0) = x_axis.at<double>(1, 0);
		v2.at<double>(2, 0) = x_axis.at<double>(2, 0);

		v3 = v1.cross(v2);
		y_axis = v3.clone();
		break;
	case 3:
		if (z_axis.at<double>(1, 0) > 0)
		{
			z_axis.at<double>(0, 0) = -1 * z_axis.at<double>(0, 0);
			z_axis.at<double>(1, 0) = -1 * z_axis.at<double>(1, 0);
			z_axis.at<double>(2, 0) = -1 * z_axis.at<double>(2, 0);
			v1.at<double>(0, 0) = z_axis.at<double>(0, 0);
			v1.at<double>(1, 0) = z_axis.at<double>(1, 0);
			v1.at<double>(2, 0) = z_axis.at<double>(2, 0);
			v2.at<double>(0, 0) = x_axis.at<double>(0, 0);
			v2.at<double>(1, 0) = x_axis.at<double>(1, 0);
			v2.at<double>(2, 0) = x_axis.at<double>(2, 0);
			v3 = v1.cross(v2);
			y_axis = v3.clone();
		}
		break;
	default:
		std::cout << ">> case default" << std::endl;
		break;
	}

	dstR.at<double>(0, 0) = x_axis.at<double>(0, 0);
	dstR.at<double>(1, 0) = x_axis.at<double>(1, 0);
	dstR.at<double>(2, 0) = x_axis.at<double>(2, 0);

	dstR.at<double>(0, 1) = y_axis.at<double>(0, 0);
	dstR.at<double>(1, 1) = y_axis.at<double>(1, 0);
	dstR.at<double>(2, 1) = y_axis.at<double>(2, 0);

	dstR.at<double>(0, 2) = z_axis.at<double>(0, 0);
	dstR.at<double>(1, 2) = z_axis.at<double>(1, 0);
	dstR.at<double>(2, 2) = z_axis.at<double>(2, 0);
}

void getRotateMat(Point2f vx, Point2f vy, double f_len, Mat &R)
{
	Mat r1 = Mat_<double>(3, 1);
	Mat r2 = Mat_<double>(3, 1);
	Mat r3 = Mat_<double>(3, 1);

	r1.at<double>(0, 0) = vx.x;
	r1.at<double>(1, 0) = vx.y;
	r1.at<double>(2, 0) = f_len;
	normalize(r1, r1);

	r2.at<double>(0, 0) = vy.x;
	r2.at<double>(1, 0) = vy.y;
	r2.at<double>(2, 0) = f_len;
	normalize(r2, r2);

	r3 = r1.cross(r2);
	R = Mat_<double>(3, 3);
	R.at<double>(0, 0) = r1.at<double>(0, 0);
	R.at<double>(1, 0) = r1.at<double>(1, 0);
	R.at<double>(2, 0) = r1.at<double>(2, 0);

	R.at<double>(0, 1) = r2.at<double>(0, 0);
	R.at<double>(1, 1) = r2.at<double>(1, 0);
	R.at<double>(2, 1) = r2.at<double>(2, 0);

	R.at<double>(0, 2) = r3.at<double>(0, 0);
	R.at<double>(1, 2) = r3.at<double>(1, 0);
	R.at<double>(2, 2) = r3.at<double>(2, 0);
	adjustRotateMat(R, R);
}

void getIntrinsicParameter(double f_len, Point2f principal_point, Mat & K)
{
	K = Mat_<double>(3, 3);
	K.at<double>(0, 0) = K.at<double>(1, 1) = f_len;
	K.at<double>(0, 1) = K.at<double>(1, 0) = K.at<double>(2, 0) = K.at<double>(2, 1) = 0;
	K.at<double>(0, 2) = principal_point.x;
	K.at<double>(1, 2) = principal_point.y;
	K.at<double>(2, 2) = 1;
}

double getF_Len(Point2f vx, Point2f vy)
{
	return sqrt(-1 * (vx.x * vy.x + vx.y * vy.y));
}

void Calibrate(Mat frame, Point2f data_point[11], double &f_len, Mat &K, Mat &R, Mat &T)
{
	assert(R.rows == 3 && R.cols == 3);
	assert(T.rows == 3 && T.cols == 1);
	Point2f point[11];
	Point2f principal_point;
	principal_point.x = (double)frame.cols / 2.0;
	principal_point.y = (double)frame.rows / 2.0;
	cout << ">> p = " << principal_point << endl;
	for (int i = 0; i<11; i++)
	{
		point[i] = data_point[i] - principal_point;
	}
	Point2f origin = point[10];

	Point2f v1 = getVanishPoint(point[0], point[1], point[2], point[3]);
	Point2f v2 = getVanishPoint(point[4], point[5], point[6], point[7]);

	f_len = getF_Len(v1, v2);
	getIntrinsicParameter(f_len, principal_point, K);
	getRotateMat(v1, v2, f_len, R);
	getTranslationVector(origin, v1, R, f_len, T);
}

