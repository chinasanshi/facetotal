//
//
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//�����ļ�ͼƬ��ʱ����Ҫ��ͷ�ļ�
//#include<conio.h>
#include "captureface.h"
#include "faceclass.h"

using namespace cv;
using namespace std;



//int main()
//{
//	//takphoto();//�ɼ����������������ͷ��ͼƬ�ļ�
//	transfer("*.jpg");//������·���µ����з������������.jpgͼƬ������ʹ��ͨ���
//
//	//Ptr<FaceRecognizer> model = createEigenFaceRecognizer(10);//����PCAģ��
//	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();//����FisherFaceģ��
//	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();//����LBPģ�ͣ���ģ�ͼ������Ч�����
//	model->train(faces, labels);//ѵ��ģ��
//	//model->set("threshold", 5200);//PCA����ֵ����
//
//	//model->save("facepca.xml");//��ѵ���õ�����PCAģ�ͱ����XML�ļ�
//	//model->load("facepca.xml");//�ӱ����XML�ļ��ж�ȡPCAģ��
//	//model->save("Fisherface.xml");//��ѵ���õ�����PCAģ�ͱ����XML�ļ�
//	//model->load("Fisherface.xml");//�ӱ����XML�ļ��ж�ȡPCAģ��
//	model->save("LBPHface1.xml");//��ѵ���õ�����PCAģ�ͱ����XML�ļ�
//	//model->load("LBPHface.xml");//�ӱ����XML�ļ��ж�ȡPCAģ��
//	//model->set("threshold", 70);//LBPH��ֵ����Ϊ80�鲻�಻�����⣻70ʱ����������������ʶ����
//	predect(model);//������ͷ��Ԥ������
//	waitKey(0);
//
//	return 0;
//}

int main()
{
	faceclass kylface;
	kylface.addcascade();
	//kylface.traversal();
	//kylface.setmodelno();
	//kylface.trainsavefacemodel();
	kylface.loadfacemodel();
	kylface.predect(true,true);
}

