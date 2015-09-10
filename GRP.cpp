
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//�����ļ�ͼƬ��ʱ����Ҫ��ͷ�ļ�

#include "sample.h"
#include "faceclass.h"

using namespace cv;
using namespace std;

int main() {
	//����ͷ����ʹ�ñ�����ģ��ÿ30֡���һ�Σ�����ʱ�����
	int flag = 3;
	faceclass kylface;
	kylface.addcascade();
	kylface.setmodelno();
	kylface.loadfacemodel();
	//������ͷ
	kylface.opencamera();
	kylface.savevideoinit();
	//ÿ30֡��Ƶ���һ�Σ�ֻ�������
	kylface.userdect(flag, true, false,true);

	////��Ƶ��ʹ�ñ�����ģ������ʾǰ��
	//int flag = 2;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////����Ƶ
	//kylface.opencamera("test.mp4",-2);
	////ÿ30֡��Ƶ���һ�Σ��������˶����
	//kylface.smartdect(flag, true, true);

	////����ͷ��ʹ�ñ�����ģ������ʾǰ��
	//int flag = 2;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////����Ƶ
	//kylface.opencamera();
	////ÿ30֡��Ƶ���һ�Σ��������˶����
	//kylface.smartdect(flag, true, true);


	////��Ƶ��ʹ�ñ�����ģ����ʾǰ��
	//int flag = 2;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////����Ƶ
	//kylface.opencamera("test.mp4", -2);
	////ÿ30֡��Ƶ���һ�Σ��������˶����
	//kylface.smartdect(flag, true, true, false, true);



	////��Ƶ����ʹ�ñ�����ģ��ÿ30֡���һ�Σ�����ʱ�����
	//int flag = 3;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////����Ƶ
	//kylface.opencamera("test.mp4", -2);
	//kylface.savevideoinit();
	////ÿ30֡��Ƶ���һ�Σ�ֻ�������
	//kylface.userdect(flag, true, true,true);

	////����ͷ����ʹ�ñ�����ģ��ÿ30֡���һ�Σ�����ʱ�����
	//int flag = 3;
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.loadfacemodel();
	////����Ƶ
	//kylface.opencamera();
	//kylface.savevideoinit();
	////ÿ30֡��Ƶ���һ�Σ�ֻ�������
	//kylface.userdect(flag, true, true);


	////ѵ���µ�����ģ��
	//faceclass kylface;
	//kylface.addcascade();
	//kylface.setmodelno();
	//kylface.train_new_model();


	system("pause");
	return 0;
}






