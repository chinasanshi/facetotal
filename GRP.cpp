
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//遍历文件图片的时候需要的头文件

#include "sample.h"
#include "faceclass.h"

using namespace cv;
using namespace std;

int main() {
	//摄像头，不使用背景建模，每30帧检测一次，其他时间跟踪
	int flag = 3;
	faceclass kylface;
	kylface.addcascade();
	kylface.setmodelno();
	kylface.loadfacemodel();
	//打开摄像头
	kylface.opencamera();
	kylface.savevideoinit();
	//每30帧视频检测一次，只检测人脸
	kylface.userdect(flag, true, false,true);

	////视频，使用背景建模，不显示前景
	//int flag = 2;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////打开视频
	//kylface.opencamera("test.mp4",-2);
	////每30帧视频检测一次，人脸行人都检测
	//kylface.smartdect(flag, true, true);

	////摄像头，使用背景建模，不显示前景
	//int flag = 2;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////打开视频
	//kylface.opencamera();
	////每30帧视频检测一次，人脸行人都检测
	//kylface.smartdect(flag, true, true);


	////视频，使用背景建模，显示前景
	//int flag = 2;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////打开视频
	//kylface.opencamera("test.mp4", -2);
	////每30帧视频检测一次，人脸行人都检测
	//kylface.smartdect(flag, true, true, false, true);



	////视频，不使用背景建模，每30帧检测一次，其他时间跟踪
	//int flag = 3;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////打开视频
	//kylface.opencamera("test.mp4", -2);
	//kylface.savevideoinit();
	////每30帧视频检测一次，只检测人脸
	//kylface.userdect(flag, true, true,true);

	////摄像头，不使用背景建模，每30帧检测一次，其他时间跟踪
	//int flag = 3;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////打开视频
	//kylface.opencamera();
	//kylface.savevideoinit();
	////每30帧视频检测一次，只检测人脸
	//kylface.userdect(flag, true, true);


	////训练新的人脸模型
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.train_new_model();


	system("pause");
	return 0;
}






