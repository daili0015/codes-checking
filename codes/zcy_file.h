#ifndef ZCYFILE_H
#define ZCYFILE_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef linux
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif
#ifdef WIN32
#include <direct.h>
#include <io.h>
#endif

using namespace std;

int mk_dir(string dir){
    if (access(dir.c_str(), 0) == -1)
    {
        cout<<dir<<" is not existing, now make it";
#ifdef WIN32
        int flag=mkdir(dir.c_str());
#endif
#ifdef linux
        int flag=mkdir(dir.c_str(), 0777);
#endif
        if (flag == 0)
        {
            cout<<"---- successfully"<<endl;
        } else {
            cout<<"---- errorly"<<endl;
        }
    }
    //    else if (access(dir.c_str(), 0) == 0)
    //    {
    //        cout<<dir<<" exists"<<endl;
    //        cout<<"now delete it"<<endl;
    //        int flag=rmdir(dir.c_str());
    //        if (flag == 0)
    //        {
    //            cout<<"delete it successfully"<<endl;
    //        } else {
    //            cout<<"delete it errorly"<<endl;
    //        }
    //    }
}

bool match(const char *pattern, const char *content) {
    if ('\0' == *pattern && '\0' == *content)
        return true;
    if ('*' == *pattern && '\0' != *(pattern + 1) && '\0' == *content)
        return false;
    if ('?' == *pattern || *pattern == *content)
        return match(pattern + 1, content + 1);
    if ('*' == *pattern)
        return match(pattern + 1, content) || match(pattern, content + 1);
    return false;
}

bool is_legal_file_type(string filepath, vector<char*> legal_file_type){
    for (auto file_type:legal_file_type){
        if(match(file_type, filepath.c_str()))
            return true;
    }
    return false;
}

bool is_dat_type(string filepath){
    if(match("*.dat", filepath.c_str()))
        return true;
    else
        return false;
}

bool is_same_file_type(string filepath, string com_filepath, vector<char*> legal_file_type){
    char file_type[20];
    for(int i=0;i<legal_file_type.size();i++){
        strcpy(file_type, legal_file_type[i]);
        strcat(file_type, ".dat");
        if(match(file_type, com_filepath.c_str())){
            if(match(file_type, filepath.c_str())){
//                cout<<filepath<<com_filepath<<endl;
                return true;
            }
            else
                return false;
        }
    }
    return false;
}

int copy_file(string from_path, string to_path){
    //    cout << "copy" << from_path << "to" << to_path  <<endl;;
    FILE *fp1;
    fp1 = fopen(from_path.c_str(), "r");
    if(fp1==NULL) return 0;
    FILE *fp2;
    fp2 = fopen(to_path.c_str(), "w");
    if(fp1==NULL) return 0;

    char buff[200] = {'\0'};
    while(fgets(buff, sizeof(buff), fp1) != NULL)
    {
        fputs(buff, fp2);
    }

    fclose(fp1);
    fclose(fp2);
    return 1;
}

int getAllFiles(string cate_dir, vector<string>& files, vector<string>& filenames)
{

    if(!match("*/", cate_dir.c_str())) cate_dir = cate_dir + "/";
    DIR *dir;
    struct dirent *ptr;
    //    cout << cate_dir<<endl;
    if ((dir=opendir(cate_dir.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8){    ///file{
            files.push_back(cate_dir+ptr->d_name);
            filenames.push_back(ptr->d_name);
        }
        else if(ptr->d_type == 4)    ///dir
        {
            //            cout << cate_dir+"/"+ptr->d_name <<endl;

            getAllFiles(cate_dir+ptr->d_name, files, filenames);
        }
    }
    closedir(dir);

    //排序，按从小到大排序
    //    sort(files.begin(), files.end());
    return 1;
}

vector<string> getFiles(string cate_dir)
{
    vector<string> files;//存放文件名

#ifdef WIN32
    _finddata_t file;
    long lf;
    //输入文件夹路径
    if ((lf=_findfirst(cate_dir.c_str(), &file)) == -1) {
        cout<<cate_dir<<" not found!!!"<<endl;
    } else {
        while(_findnext(lf, &file) == 0) {
            //输出文件名
            //cout<<file.name<<endl;
            if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
                continue;
            files.push_back(file.name);
        }
    }
    _findclose(lf);
#endif

#ifdef linux
    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(cate_dir.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            files.push_back(ptr->d_name);
        else if(ptr->d_type == 10)    ///link file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            continue;
        else if(ptr->d_type == 4)    ///dir
        {
            files.push_back(ptr->d_name);
            /*
                memset(base,'\0',sizeof(base));
                strcpy(base,basePath);
                strcat(base,"/");
                strcat(base,ptr->d_nSame);
                readFileList(base);
            */
        }
    }
    closedir(dir);
#endif

    //排序，按从小到大排序
    sort(files.begin(), files.end());
    return files;
}

vector<string> getDirs(string cate_dir)
{
    vector<string> files;//存放文件名

#ifdef WIN32
    _finddata_t file;
    long lf;
    //输入文件夹路径
    if ((lf=_findfirst(cate_dir.c_str(), &file)) == -1) {
        cout<<cate_dir<<" not found!!!"<<endl;
    } else {
        while(_findnext(lf, &file) == 0) {
            //输出文件名
            //cout<<file.name<<endl;
            if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
                continue;
            files.push_back(file.name);
        }
    }
    _findclose(lf);
#endif

#ifdef linux
    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(cate_dir.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            continue;
        else if(ptr->d_type == 10)    ///link file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            continue;
        else if(ptr->d_type == 4)    ///dir
        {
            files.push_back(ptr->d_name);
        }
    }
    closedir(dir);
#endif

    //排序，按从小到大排序
    sort(files.begin(), files.end());
    return files;
}

#endif //
