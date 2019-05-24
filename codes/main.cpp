#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <sys/time.h>
#include "zcy_sift.h"
#include "code2img.h"
#include "zcy_compare.h"
#include "zcy_file.h"
#include "sift_stream.h"
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;
using namespace zcy_sift_lib;

int main(int args, char *argv[])
{
    string mode = argv[1];
    vector<char*> legal_file_type = {"*.m",  "*.c", "*.cpp", "*.py", "*.h", "*.hpp"};
    if(mode=="-c"){
        string filepath1 = argv[2];
        string filepath2 = argv[3];
        cout << "Compare similarity between " << filepath1 << " and " << filepath2 <<endl;
        Mat ResImg1, ResImg2;
        if(!code2img(filepath1, ResImg1)) cout<<"filepath1 code2img error"<<endl;
        if(!code2img(filepath2, ResImg2)) cout<<"filepath2 code2img error"<<endl;
        if(!zcy_compare(filepath1+".png", filepath2+".png")) cout<<"zcy_compare error"<<endl;
    }else if(mode=="-codes2imgs"){
        string dirpath1 = argv[2];
        string dirpath2 = argv[3];
        cout << "Generate imgs from " << dirpath1 << " to " << dirpath2 <<endl;
        vector<string> Dirs=getDirs(dirpath1);
        zcy_sift my_sift = zcy_sift();
        sift_stream my_stream = sift_stream();
        for (int i=0; i<Dirs.size(); i++)
        {
            mk_dir(dirpath2+"/"+Dirs[i]);
            vector<string> files, filenames;
            getAllFiles(dirpath1+"/"+Dirs[i]+"/", files, filenames);
            for(int k=0;k<files.size();k++){
                if(is_legal_file_type(files[k], legal_file_type)){
                    if(!copy_file(files[k], dirpath2+"/"+Dirs[i]+"/"+filenames[k])){
                        cout << "copy" << files[k] << "to" << dirpath2+"/"+Dirs[i]+"/"+filenames[k]  <<endl;
                        continue;
                    }
                    Mat ResImg;
                    if(!code2img(files[k], ResImg, dirpath2+"/"+Dirs[i])){
                        cout<<files[k]<<"   code2img error"<<endl;
                        continue;
                    }
                    //compute sift data and save
                    vector<KeyPoint> sift_Keypoint;
                    my_sift.detect(ResImg, sift_Keypoint);
                    Mat sift_Descriptor;
                    my_sift.compute(ResImg,sift_Keypoint,sift_Descriptor);
                    //save sift data
                    my_stream.save_sift(dirpath2+"/"+Dirs[i]+"/"+filenames[k]+".dat",sift_Keypoint,sift_Descriptor);
                }
            }
        }
    }else if(mode=="-CheckCurDir"){
        string dirpath = argv[2];
        cout << "Check Current Dir's codes " << dirpath << endl;
        vector<string> Dirs=getDirs(dirpath);
        for (int i=0; i<Dirs.size(); i++)
        {
            vector<string> files, filenames;
            vector<string> compare_Dirs = Dirs;
            compare_Dirs.erase(compare_Dirs.begin()+i);
            getAllFiles(dirpath+"/"+Dirs[i]+"/", files, filenames);
            for(int k=0;k<files.size();k++){
                if(is_dat_type(files[k])){
//                    cout << files[k] << endl;
                    zcy_com_with_dir(files[k], dirpath, compare_Dirs);
                }
            }
        }
    }else if(mode=="-h"){
        cout << "usages 1: Compare similarity between two codes" <<endl;
        cout << "          -c /root/codes/main.cpp /root/codes/test.cpp" <<endl;
        cout << "usages 2: Generate sift images directory for all folders in the directory" <<endl;
        cout << "          -codes2imgs /root/CPP/codes /root/CPP/sift_imgs" <<endl;
        cout << "usages 3: Check sift images directory " <<endl;
        cout << "          -CheckCurDir /root/CPP/sift_imgs" <<endl;
    }
}

