TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH += /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2

LIBS += /usr/local/lib/libopencv_highgui.so  \
        /usr/local/lib/libopencv_core.so  \
        /usr/local/lib/libopencv_imgproc.so  \
        /usr/local/lib/libopencv_imgcodecs.so  \
        /usr/local/lib/libopencv_features2d.so  \
#        /usr/local/lib/libopencv_xfeatures2d.so \
        /usr/local/lib/libopencv_calib3d.so \
#        /usr/lib/x86_64-linux-gnu/libboost_filesystem.so \
#        /usr/local/lib/libopencv_face.so \
#        /usr/local/lib/libopencv_objdetect.so \

HEADERS += \
    zcy_sift.h \
    code2img.h \
    zcy_file.h \
    sift_stream.h \
    zcy_compare.h

