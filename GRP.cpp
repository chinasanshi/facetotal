//
//
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//遍历文件图片的时候需要的头文件
//#include<conio.h>
#include "sample.h"
#include "faceclass.h"

using namespace cv;
using namespace std;



//int main()
//{
//	//takphoto();//采集人脸样本，保存成头像图片文件
//	transfer("*.jpg");//遍历该路径下的所有符合命名规则的.jpg图片，可以使用通配符
//
//	//Ptr<FaceRecognizer> model = createEigenFaceRecognizer(10);//定义PCA模型
//	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();//定义FisherFace模型
//	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();//定义LBP模型，此模型检测人脸效果最好
//	model->train(faces, labels);//训练模型
//	//model->set("threshold", 5200);//PCA的阈值设置
//
//	//model->save("facepca.xml");//将训练好的人脸PCA模型保存成XML文件
//	//model->load("facepca.xml");//从保存的XML文件中读取PCA模型
//	//model->save("Fisherface.xml");//将训练好的人脸PCA模型保存成XML文件
//	//model->load("Fisherface.xml");//从保存的XML文件中读取PCA模型
//	model->save("LBPHface1.xml");//将训练好的人脸PCA模型保存成XML文件
//	//model->load("LBPHface.xml");//从保存的XML文件中读取PCA模型
//	//model->set("threshold", 70);//LBPH阈值设置为80查不多不会误检测；70时脸必须是正脸否则识别不了
//	predect(model);//打开摄像头，预测人脸
//	waitKey(0);
//
//	return 0;
//}

int main()
{
	//sample kylsample;
	//kylsample.addcascade();
	//kylsample.runvedio("nothing");
	//kylsample.takephoto("11", "Ann", "1");//diaoyong


	faceclass kylface;
	kylface.addcascade();
	kylface.setmodelno();
	kylface.loadfacemodel();

	kylface.runvedio("nothing");//diaoyong

	//kylface.train_new_model();//diaoyong
	
	kylface.smartdect(true,true);//diaoyong
	kylface.userdect(true,true);//diaoyong

	return 0;
}

