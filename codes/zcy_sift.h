#ifndef ZCY_SIFT_H
#define ZCY_SIFT_H

#include <iostream>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace cv
{
namespace zcy_sift_lib
{

class zcy_sift
{
public:
    static Ptr<zcy_sift> create( int nfeatures = 0, int nOctaveLayers = 3,
                                 double contrastThreshold = 0.04, double edgeThreshold = 10,
                                 double sigma = 1.6);
    explicit zcy_sift( int nfeatures = 0, int nOctaveLayers = 3,
                       double contrastThreshold = 0.04, double edgeThreshold = 10,
                       double sigma = 1.6);

    //! returns the descriptor size in floats (128)
    //! //返回描述符维度
    int descriptorSize() const;

    //! returns the descriptor type
    //! //返回描述符类型
    int descriptorType() const;

    //! returns the default norm type
    int defaultNorm() const;

    //! finds the keypoints and computes descriptors for them using SIFT algorithm.
    //! Optionally it can compute descriptors for the user-provided keypoints
    void detectAndCompute(InputArray img, InputArray mask,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray descriptors,
                          bool useProvidedKeypoints = false);

    void buildGaussianPyramid( const Mat& base, std::vector<Mat>& pyr, int nOctaves ) const;
    void buildDoGPyramid( const std::vector<Mat>& pyr, std::vector<Mat>& dogpyr ) const;
    void findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                                std::vector<KeyPoint>& keypoints ) const;
    void compute( InputArray image, std::vector<KeyPoint>& keypoints,
                  OutputArray descriptors );
    void detect( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                 InputArray mask=noArray() );
protected:
    CV_PROP_RW int nfeatures;
    CV_PROP_RW int nOctaveLayers;
    CV_PROP_RW double contrastThreshold;
    CV_PROP_RW double edgeThreshold;
    CV_PROP_RW double sigma;
};

/******************************* Defs and macros *****************************/

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#if 0
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    //int octv=kpt.octave & 255, layer=(kpt.octave >> 8) & 255;   //该特征点所在的组序号和层序号
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);//缩放倍数
}

static Mat createInitialImage( const Mat& img, bool doubleImageSize, float sigma )
{
//    std::cout<<"doubleImageSize"<<doubleImageSize<<std::endl;
    Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
        cvtColor(img, gray, COLOR_BGR2GRAY);//原始图像转灰度
    else
        img.copyTo(gray);
    //缩放并转换到另外一种数据类型,深度转换为CV_16S避免外溢。（48,0）为缩放参数
    //灰度值拉伸了48倍,CV_16S避免外溢
    gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;
    if( doubleImageSize )
    {
        //sigma=1.6,SIFT_INIT_SIGMA=0.5
        // SIFT_INIT_SIGMA 为 0.5,即输入图像的尺度,SIFT_INIT_SIGMA×2=1.0,即图像扩
        //大 2 倍以后的尺度,sig_diff 为公式 4 中的高斯函数所需要的方差
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        Mat dbl;
        //利用双线性插值法把图像的长宽都扩大 2 倍
        resize(gray_fpt, dbl, Size(gray.cols*2, gray.rows*2), 0, 0, INTER_LINEAR);
        //利用公式 3 对图像进行高斯平滑处理
        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        return dbl;
    }
    else //不需要扩大图像的尺寸
    {
        // sig_diff 为公式 4 中的高斯函数所需要的方差
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        return gray_fpt;
    }
}

Ptr<zcy_sift> zcy_sift::create( int _nfeatures, int _nOctaveLayers,
                                double _contrastThreshold, double _edgeThreshold, double _sigma )
{
    return makePtr<zcy_sift>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma);
}

void zcy_sift::compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) {
    if( image.empty() )
    {
        descriptors.release();
        return;
    }
    detectAndCompute(image, noArray(), keypoints, descriptors, true);
}
void zcy_sift::detect( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints, InputArray mask){
    if( image.empty() )
    {
        keypoints.clear();
        return;
    }
    detectAndCompute(image, mask, keypoints, noArray(), false);
}

void zcy_sift::buildGaussianPyramid( const Mat& base, std::vector<Mat>& pyr, int nOctaves ) const
{
    //向量数组 sig 表示每组中计算各层图像所需的方差,nOctaveLayers + 3 即为公式 7
    std::vector<double> sig(nOctaveLayers + 3);
    // 构建nOctaves组（每组nOctaveLayers+3层）高斯金字塔
    //定义高斯金字塔的总层数,nOctaves*(nOctaveLayers + 3)即组数×层数
    pyr.resize(nOctaves*(nOctaveLayers + 3));//pyr保存所有组所有层

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    //计算第0组每层的尺度因子sig[i]，第0组第0层已经模糊过，所有只有5层需要模糊
    //提前计算好各层图像所需的方差
    sig[0] = sigma;  //第一层图像的尺度为基准层尺度 σ 0
    double k = std::pow( 2., 1. / nOctaveLayers );//由公式 8 计算 k 值
    //遍历所有层,计算方差
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*sigma;//由公式 10 计算前一层图像的尺度
        double sig_total = sig_prev*k;//由公式 10 计算当前层图像的尺度
        //计算公式 4 中高斯函数所需的方差,并存入 sig 数组内
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    //512大小的图像，nOctaves=7;
    ////遍历高斯金字塔的所有层,构建高斯金字塔
    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ )
        {
            Mat& dst = pyr[o*(nOctaveLayers + 3) + i];//dst 为当前层图像矩阵
            //第0组第0层为base层，即原始图像 //如果当前层为高斯金字塔的第 0 组第 0 层,则直接赋值
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            //高斯金字塔的新组(new octave)的第0幅为上一组的第nOctaveLayers幅下采样得到，采样步长为2
            //如果当前层是除了第 0 组以外的其他组中的第 0 层,则要进行降采样处理
            else if( i == 0 )
            {
                const Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols/2, src.rows/2),
                       0, 0, INTER_NEAREST);//隔点降采样处理
            }
            // 每一组的第i幅图像是由该组第i-1幅图像用sig[i]高斯模糊得到，相当于使用了新的尺度。
            else
            {
                const Mat& src = pyr[o*(nOctaveLayers + 3) + i-1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]);//根据公式 3,由前一层尺度图像得到当前层的尺度图像
            }
        }
    }
}


void zcy_sift::buildDoGPyramid( const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr ) const
{
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3); //nOctaves表示组的个数
    dogpyr.resize( nOctaves*(nOctaveLayers + 2) ); //保存所有组的Dog图像
    //每组相邻两幅图像相减，获取Dog图像
    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 2; i++ )
        {
            const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
            const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
            Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
            //两幅图像相减
            subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
        }
    }
}


// Computes a gradient orientation histogram at a specified pixel
//计算某一个特征点的周围区域梯度方向直方图
static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n )
{
    //len表示以2*radius+1为半径的圆(因为点是离散的，其实为正方形)的像素个数
    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    AutoBuffer<float> buf(len*4 + n+4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true); //计算向量角度
    cv::hal::magnitude32f(X, Y, Mag, len); //计算梯度

    for( k = 0; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    //平滑
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];
    for( i = 0; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
                (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
                temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval; //返回直方图中最大值
}


// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
//  dog_pyr:dog金字塔;
//  kpt:关键点;
//  octv:组序号
//  layer: dog层序号
//  r: 行号; c:列号
//  nOctaveLayers:dog中要用到的层数，为3
//  contrastThreshold:对比度阈值=0.04
//   edgeThreshold:边界阈值=10
//  sigma: 尺度因子
static bool adjustLocalExtrema( const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
                                int& layer, int& r, int& c, int nOctaveLayers,
                                float contrastThreshold, float edgeThreshold, float sigma )
{
    //表示灰度归一化要用到的参数
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);//金字塔图像灰度拉伸了48倍，所以要除48.SIFT_FIXPT_SCALE=48
    const float deriv_scale = img_scale*0.5f;//一阶导数灰度归一化参数
    const float second_deriv_scale = img_scale; //dxx，dyy。二阶导数灰度归一化参数
    const float cross_deriv_scale = img_scale*0.25f;//dxy，交叉导数灰度归一化参数

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;
    //循环5次，对坐标进行5次修正。
    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer; //得到该特征点所在的dog层序号
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        //计算一阶偏导数，并归一化，通过临近点差分求得
        Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        //计算二阶偏导数，并归一化，通过临近点差分求得
        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        //二阶偏导数矩阵
        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        //令尺度方程的泰勒展开导数为0，求解出偏移量X
        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];      //层偏移，层偏移即尺度偏移
        xr = -X[1];     //行偏移
        xc = -X[0];    //列偏移

        //如果求解出的偏移量均小于0.5，则退出循环，说明该关键点的选取是正确的
        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
                std::abs(xr) > (float)(INT_MAX/3) ||
                std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        //求解出的偏移量均大于0.5，则将原坐标加上求出来的偏移量，得到更精确的坐标和尺度
        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        //层数或行列超出边界则退出
        if( layer < 1 || layer > nOctaveLayers ||
                c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
                r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    //保证插值收敛，SIFT_MAX_INTERP_STEPS=5
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;

    {
        //以下代码用来去除低对比度的不稳定特征点（灰度要归一化）
        int idx = octv*(nOctaveLayers+2) + layer; //求出修正后的层序号
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        //求出修正后的关键点坐标的一阶微分向量
        Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        //一阶偏导向量与上面求出的偏移量向量求点积
        float t = dD.dot(Matx31f(xc, xr, xi));
        //修正后的坐标带入泰勒展开式，得到的结果小于contrastThreshold=0.04则抛弃该点
        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
            return false;

        // principal curvatures are computed using the trace and det of Hessian
        //利用Hessian矩阵的迹和行列式计算该关键点的主曲率的比值
        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        //edgeThreshold=10,去除边缘点
        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
            return false;
    }

    //精确特征点在原图像上的位置
    kpt.pt.x = (c + xc) * (1 << octv); //高斯金字塔坐标根据组数扩大相应的倍数
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16); //特征点被检测出时所在的金字塔组
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2; //关键点邻域直径
    kpt.response = std::abs(contr);

    return true;
}


//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void zcy_sift::findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                                      std::vector<KeyPoint>& keypoints ) const
{
    int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);//组的个数
    // The contrast threshold used to filter out weak features in semi-uniform
    // (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    //低 对比度的阈值， contrastThreshold默认为0.04
    int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    const int n = SIFT_ORI_HIST_BINS;  //直方图bin的个数=36，每个10度 定义梯度方向的数量
    float hist[n];
    KeyPoint kpt;

    keypoints.clear();

    for( int o = 0; o < nOctaves; o++ )
        for( int i = 1; i <= nOctaveLayers; i++ ) // nOctaveLayers表示每层Dog图像个数-2，即最终用到的层数
        {
            int idx = o*(nOctaveLayers+2)+i;
            const Mat& img = dog_pyr[idx]; //获取该层Dog图像，序号从1开始。第0层和最后一层不用
            const Mat& prev = dog_pyr[idx-1]; //上一幅Dog图像
            const Mat& next = dog_pyr[idx+1]; //下一幅Dog图像
            int step = (int)img.step1();
            int rows = img.rows, cols = img.cols;

            //SIFT_IMG_BORDER=5，边界5个像素的距离
            for( int r = SIFT_IMG_BORDER; r < rows-SIFT_IMG_BORDER; r++)
            {
                //获取3幅相邻的Dog图像的行指针
                const sift_wt* currptr = img.ptr<sift_wt>(r);//DoG 金字塔当前层图像的当前行指针
                const sift_wt* prevptr = prev.ptr<sift_wt>(r);
                const sift_wt* nextptr = next.ptr<sift_wt>(r);

                for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
                {
                    //获取该层图像r行c列的像素值
                    //DoG 金字塔当前层尺度图像的像素值
                    sift_wt val = currptr[c];

                    // find local extrema with pixel accuracy
                    //与周围26个点比较，极大或极小值则为局部极值点
                    //精确定位局部极值点
                    //如果满足 if 条件,则找到了极值点,即候选特征点
                    if( std::abs(val) > threshold &&//像素值要大于一定的阈值才稳定,
                            //即要具有较强的对比度
                            //下面的逻辑判断被“与”分为两个部分,前一个部分要满足像素
                            //值大于 0,在 3×3×3 的立方体内与周围 26 个邻近像素比较找极大值,后一个部分要满足像
                            //素值小于 0,找极小值
                            ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
                              val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
                              val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
                              val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
                              val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
                              val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
                              val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
                              val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
                              val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
                             (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
                              val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
                              val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
                              val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
                              val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
                              val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
                              val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
                              val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
                              val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
                    {
                        //找到极值点之后，在Dog中调整局部极值点
                        //调整后的关键点具有以下三个属性
                        //在原图中的精确坐标，
                        //特征点所在的高斯金字塔组，即更精确的o
                        //领域直径
                        int r1 = r, c1 = c, layer = i;
                        //如果满足 if 条件,说明该极值点不是特征点,继续上面的 for 循环
                        if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                                nOctaveLayers, (float)contrastThreshold,
                                                (float)edgeThreshold, (float)sigma) )
                            continue;
                        float scl_octv = kpt.size*0.5f/(1 << o);   //获取该特征点的尺度
                        //calcOrientationHist计算该特征点周围的方向直方图，并返回直方图最大值
                        //参数o和c1,r1均已经经过精确定位
                        //方向直方图的计算是在该点尺度的高斯金字塔图像中计算的，不是在Dog图像，也不是在原图
                        float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                Point(c1, r1), //特征点坐标
                                cvRound(SIFT_ORI_RADIUS * scl_octv),//直方图统计半径：3*1.5*σ，SIFT_ORI_RADIUS=3*1.5
                                SIFT_ORI_SIG_FCTR * scl_octv,//直方图平滑所用到的尺度，SIFT_ORI_SIG_FCTR=1.5f
                                hist, n);
                        float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);//辅方向为0.8*主方向最大值，SIFT_ORI_PEAK_RATIO=0.8f
                        //计算特征点的方向
                        for( int j = 0; j < n; j++ ) //n=36
                        {
                            //j 为直方图当前柱体索引,l 为前一个柱体索引,r2 为后一
                            //个柱体索引,如果 l 和 r2 超出了柱体范围,则要进行圆周循环处理
                            int l = j > 0 ? j - 1 : n - 1;
                            int r2 = j < n-1 ? j + 1 : 0;
                            //方向角度拟合处理
                            //判断柱体高度是否大于直方图辅方向的阈值,因为拟合处
                            //理的需要,还要满足柱体的高度大于其前后相邻两个柱体的高度
                            if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                            {
                                float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;//圆周循环处理
                                kpt.angle = 360.f - (float)((360.f/n) * bin);//得到关键点的方向
                                if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)//如果方向角度十分接近于 360 度,则就让它等于 0 度
                                    kpt.angle = 0.f;
                                keypoints.push_back(kpt); //这里保存的特征点具有位置，尺度和方向3个信息
                            }
                        }
                    }
                }
            }
        }
}


static void calcSIFTDescriptor( const Mat& img, Point2f ptf, float ori, float scl,
                                int d, int n, float* dst )
{
    //d = SIFT_DESCR_WIDTH=4，描述直方图的宽度
    //n = SIFT_DESCR_HIST_BINS=8

    Point pt(cvRound(ptf.x), cvRound(ptf.y));  //坐标点取整
    float cos_t = cosf(ori*(float)(CV_PI/180));  //余弦值
    float sin_t = sinf(ori*(float)(CV_PI/180));   //正弦值
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt(((double) img.cols)*img.cols + ((double) img.rows)*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;

    AutoBuffer<float> buf(len*6 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

     //初始化直方图
    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                    r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }

    len = k;
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);
    cv::hal::exp32f(W, W, len);

    for( k = 0; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                dst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    for( k = 0; k < len; k++ )
        nrm2 += dst[k]*dst[k];
    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    for( i = 0, nrm2 = 0; i < k; i++ )
    {
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
    for( k = 0; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
    }
#else
    float nrm1 = 0;
    for( k = 0; k < len; k++ )
    {
        dst[k] *= nrm2;
        nrm1 += dst[k];
    }
    nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
    for( k = 0; k < len; k++ )
    {
        dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
    }
#endif
}

static void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
                            Mat& descriptors, int nOctaveLayers, int firstOctave )
{
    //SIFT_DESCR_WIDTH=4，描述直方图的宽度
    //SIFT_DESCR_HIST_BINS=8
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

    for( size_t i = 0; i < keypoints.size(); i++ )
    {
        KeyPoint kpt = keypoints[i];
        int octave, layer;
        float scale;
        unpackOctave(kpt, octave, layer, scale);
        CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);
        float size=kpt.size*scale; //该特征点所在组的图像尺寸
        Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale); //该特征点在金字塔组中的坐标
        const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];//该点所在的金字塔图像

        float angle = 360.f - kpt.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;
        calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

zcy_sift::zcy_sift( int _nfeatures, int _nOctaveLayers,
                    double _contrastThreshold, double _edgeThreshold, double _sigma )
    : nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
      contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
{
}
// sigma：对第0层进行高斯模糊的尺度空间因子。
// 默认为1.6（如果是软镜摄像头捕获的图像，可以适当减小此值）


int zcy_sift::descriptorSize() const
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int zcy_sift::descriptorType() const
{
    return CV_32F;
}

int zcy_sift::defaultNorm() const
{
    return NORM_L2;
}


void zcy_sift::detectAndCompute(InputArray _image, InputArray _mask,
                                std::vector<KeyPoint>& keypoints,
                                OutputArray _descriptors,
                                bool useProvidedKeypoints)
{
    //detectAndCompute(image, mask, keypoints, noArray(), false);
    //detectAndCompute(image, noArray(), keypoints, descriptors, true);

    //firstOctave 表示金字塔的组索引是从 0 开始还是从‐1 开始,‐1 表示需要对输入图像的
    //长宽扩大一倍,actualNOctaves 和 actualNLayers 分别表示实际的金字塔的组数和层数
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image = _image.getMat(), mask = _mask.getMat();

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

    if( useProvidedKeypoints )
    {
        //因为不需要扩大输入图像的长宽,所以重新赋值 firstOctave 为 0
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            int octave, layer;
            float scale;
            //从输入的特征点变量中提取出该特征点所在的组、层、以及它的尺度
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        } 
        firstOctave = std::min(firstOctave, 0);//确保最小组索引号不大于 0
        //确保最小组索引号大于等于‐1,实际层数小于等于输入参数 nOctaveLayers
        CV_Assert( firstOctave >= -1 && actualNLayers <= nOctaveLayers );
        //计算实际的组数
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    // 得到第1组（Octave）图像
    //创建基层图像矩阵 base,详见下面对 createInitialImage 函数的分析
    //createInitialImage 函数的第二个参数表示是否进行扩大输入图像长宽尺寸操作,true
    //表示进行该操作,第三个参数为基准层尺度 σ 0
    Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
    std::vector<Mat> gpyr, dogpyr;
    // 每层金字塔图像的组数（Octave）
    // 计 算 金 字 塔 的 组 的 数 量 , 当 actualNOctaves  >  0 时 , 表 示 进 入 了 上 面 的
    //if( useProvidedKeypoints )语句,所以组数直接等于 if( useProvidedKeypoints )内计算得到的值
    //如果 actualNOctaves 不大于 0,则利用公式 6 计算组数
    //这里面还考虑了组的初始索引等于‐1 的情况,所以最后加上了  – firstOctave 这项
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log( (double)std::min( base.cols, base.rows ) ) / std::log(2.) - 2) - firstOctave;

    // 构建金字塔（金字塔层数和组数相等）
    buildGaussianPyramid(base, gpyr, nOctaves);
    // 构建高斯差分金字塔
    buildDoGPyramid(gpyr, dogpyr);

//    for(int i=0;i<5;i++){
//        imshow(std::to_string(i), dogpyr[i]);

//    }
//    waitKey(0);

    // useProvidedKeypoints默认为false
    // 使用keypoints并计算特征点的描述符
    if( !useProvidedKeypoints )
    {
        //找到特征点并去除重复特征点
        findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
        //在特征点检测的过程中(尤其是在泰勒级数拟合的过程中)会出现特征点被重复
        //检测到的现象,因此要剔除掉那些重复的特征点
        KeyPointsFilter::removeDuplicated( keypoints );

        //保留指定数目的关键点
        // retainBest:根据相应保留指定数目的特征点（features2d.hpp）
        if( nfeatures > 0 )
            KeyPointsFilter::retainBest(keypoints, nfeatures);
        //如果 firstOctave < 0,则表示对输入图像进行了扩大处理,所以要对特征点的一些
        //变量进行适当调整。这是因为 firstOctave  <  0,金字塔增加了一个第‐1 组,而在检测特征点
        //的时候,所有变量都是基于这个第‐1 组的基准层尺度图像的。
        if( firstOctave < 0 )
            for( size_t i = 0; i < keypoints.size(); i++ )
            {
                KeyPoint& kpt = keypoints[i];
                float scale = 1.f/(float)(1 << -firstOctave);//其实这里的 firstOctave = ‐1,所以 scale = 0.5
                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                kpt.pt *= scale;
                kpt.size *= scale;
            }

         // mask标记检测区域（可选）
        if( !mask.empty() )
            KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }
    else
    {
        // filter keypoints by mask
        //KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    // 特征点输出数组
    if( _descriptors.needed() )
    {
        //计算描述符
        //t = (double)getTickCount();
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, CV_32F);
        Mat descriptors = _descriptors.getMat();

        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
        //t = (double)getTickCount() - t;
        //printf("descriptor extraction time: %g\n", t*1000./tf);
    }
}

}
}

#endif // ZCY_SIFT_H
