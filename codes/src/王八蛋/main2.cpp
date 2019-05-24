#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
 #include <sstream>



vector<vector<int>> get_flow_xy(string file_path){
    ifstream infile;
    infile.open(file_path);
    if(!infile) cout<<"error"<<endl;
    cout<<"存入vector"<<endl;
    string str;
    vector<int> position_vect;
    vector<vector<int>> flow_vect_xy;
    flow_vect_xy.clear();
    while(getline(infile,str))             //按空格读取，遇到空白符结束
    {
        position_vect = explode(str, ' ');
        flow_vect_xy.push_back(position_vect);
    }
    infile.close();
    return flow_vect_xy;
}

Mat comMatR(Mat Matrix1,Mat Matrix2,Mat &MatrixCom)
{
    CV_Assert(Matrix1.rows==Matrix2.rows);//行数不相等，出现错误中断
    MatrixCom.create(Matrix1.rows,Matrix1.cols+Matrix2.cols,Matrix1.type());
    Mat temp=MatrixCom.colRange(0,Matrix1.cols);
    Matrix1.copyTo(temp);
    Mat temp1=MatrixCom.colRange(Matrix1.cols,Matrix1.cols+Matrix2.cols);
    Matrix2.copyTo(temp1);
    return MatrixCom;
}


vector<int> explode(const string& s, const char& c)
{
    string buff{""};
    int position[2];
    vector<string> v;
    vector<int> position_vect;
    for(auto n:s)
    {
        if(n != c) buff+=n;
        else if(n == c && buff != "") { v.push_back(buff); buff = ""; }
    }
    if(buff != "") v.push_back(buff);

    stringstream stream1(v[0]);
    stream1>>position[0];
    stringstream stream2(v[1]);
    stream2>>position[1];
    position_vect.push_back(position[0]);
    position_vect.push_back(position[1]);
    return position_vect;
}

int main()
{

//    Mat src_img = imread("/home/DIA/Deep-Image-Analogy-linux/deep_image_analogy/demo/style.png");
//    Mat res_img = imread("/home/DIA/Deep-Image-Analogy-linux/deep_image_analogy/demo/output/resultAB3-3.png");

    Mat src_img = imread("/home/DIA/Deep-Image-Analogy-linux/deep_image_analogy/demo/src_tgt_cut.png");
    Mat res_img = imread("/home/DIA/Deep-Image-Analogy-linux/deep_image_analogy/demo/output/resultAB.png");

    vector<vector<int>> flow_vect_xy = get_flow_xy("/home/DIA/Deep-Image-Analogy-linux/deep_image_analogy/demo/output/flowAB.txt");

    Mat map_img;
//    comMatR(src_img, res_img, map_img);
    comMatR(res_img, src_img, map_img);

    int row_point_num = 100;
    int col_point_num = 100;
    vector<int> row_position, col_position;
    for(int i=1;i<=row_point_num;i++){
        row_position.push_back(int(src_img.rows/(row_point_num+1)*i));
    }
    for(int i=1;i<=col_point_num;i++){
        col_position.push_back(int(src_img.cols/(col_point_num+1)*i));
    }

    for(int i=0;i<row_point_num;i++){
        for(int j=0;j<col_point_num;j++){
            if(res_img.at<cv::Vec3b>(row_position[i], col_position[j])[0]>0.0){
//            if(true){

//                if(row_position[i]<330) continue;
//                if(row_position[i]>380) continue;
//                if(col_position[j]<320) continue;
//                if(col_position[j]>380) continue;
                vector<int> flow_xy = flow_vect_xy[row_position[i]*src_img.cols+col_position[j]];
//                vector<int> flow_xy = flow_vect_xy[row_position[i]*src_img.rows+col_position[j]];
//                vector<int> flow_xy = flow_vect_xy[col_position[i]*src_img.cols+row_position[j]];
//                vector<int> flow_xy = flow_vect_xy[col_position[i]*src_img.rows+row_position[j]];
                line(map_img,Point(col_position[j], row_position[i]),
                     Point(col_position[j] + src_img.cols + flow_xy[0],
                           row_position[i] + flow_xy[1]),Scalar(0,0,255),1,CV_AA);
            }
        }
    }

    imshow("map_img",map_img);
    waitKey(0);

    return 0;

}

//            putText(map_img, to_string(++count)+" "+to_string(i)+to_string(j), Point(col_position[j],
//                    row_position[i]), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0),1,4);






