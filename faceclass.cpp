//
//
//#include <windows.h>
#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//�����ļ�ͼƬ��ʱ����Ҫ��ͷ�ļ�
#include <map>
#include "faceclass.h"

using namespace cv;
using namespace std;

faceclass::faceclass(){}
faceclass::~faceclass(){}

//faceclass.hͷ�ļ�����cascade_name��Ĭ��ֵ
bool faceclass::addcascade(char* cascade_name)
{
	_cascade_name = cascade_name;

	if (!_face_cascade.load(_cascade_name))//�ж�Haar���������Ƿ�ɹ�
	{
		std::cout << "�޷����ؼ����������ļ���" << std::endl;
		return false;
	}
	else
	{
		std::cout << "�ɹ����ؼ���������" << cascade_name << std::endl;
		return true;
	}
}

void faceclass::guassforeground(cv::Mat& image, double learningspeed, bool showforeground)
//�ڶ��������������ĸ������ʣ��������������������Ƿ���ʾ������ǰ��
{
	// �˶�ǰ����⣬�����±���;_mog�Ƕ�����faceclass�е�һ����ȡ������BackgroundSubtractorMOG2�Ķ���
	_mog(image, _foreground, learningspeed);//learningspeedΪ�������ʣ�Ĭ��Ϊ0.05�����Լ�����
	//ȥ������
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);//����
	erode(_foreground, _foreground, Mat(), Point(-1, -1), 2);//��ʴ
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);

	//_mog.getBackgroundImage(_background);   // ���ص�ǰ����ͼ��

	_showforeground = showforeground;
	if (_showforeground && (!_foreground.empty()))
	{
		imshow("foreground", _foreground);//��ʾǰ��
		//imshow("background", _background);//��ʾ����
	}
}

void faceclass::getforegroundrect(bool showforegroundrect)//�������������Ƿ���ʾǰ���˶�������ο�ֻ����ǰ����ʾ��ͬʱ��Ч
{
	cv::Mat fgdrect;
	_foreground.copyTo(fgdrect);//����ǰ�������ڼ������
	fgdrect = fgdrect > 50;//�������ش���50�����ػ���Ϊ255����������Ϊ0
	//���ǰ��������
	//findContours(fgdrect, _contours, _hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//ֻ��������������
	//findContours(fgdrect, _contours, _hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//���������������Ϊ����
	findContours(fgdrect, _contours, _hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//��������������ع����

	//_foregroundrects�洢������������Ӿ��Σ���boundingRect��������
	int idx = 0;//��������ѭ��
	if (_contours.size())//������ϴ��жϣ�������Ƶ��ֻ�б���ʱ�����
	{
		for (; idx >= 0; idx = _hierarchy[idx][0])//�ҵ��������������hierarchy[idx][0]��ָ����һ����������û����һ��������hierarchy[idx][0]Ϊ������
		{
			if (fabs(contourArea(Mat(_contours[idx]))) > 5000)//�����ǰ������������ڴ�ǰ���������ֵ���򱣴浱ǰֵ
			{
				_foregroundrects.push_back(boundingRect(_contours[idx]));//ѹջ���������������Ӿ���
			}
		}
	}

	if (_showforeground && showforegroundrect)//�����ʾǰ�������������Ƿ���ʾǰ���˶��ľ��ο�
	{
		for (vector<Rect>::iterator it = _foregroundrects.begin(); it != _foregroundrects.end(); it++)//�������з�����������Ӿ���
		{
			rectangle(_foreground, *it, Scalar(255, 0, 255), 3, 8, 0);//��ǰ������������������Ӿ��ο�
		}
	}
}

void faceclass::pedestriandect(cv::Mat& image, bool showpedestrianrect)
{
	_allpedestrianrects.clear();//���¼����Ҫ���
	HOGDescriptor hog;//����HOG���󣬲���Ĭ�ϲ���
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//����Ĭ�ϵ����˼�������
	vector<Rect> pedestrianrects_filtered;//�洢������˵ı߽�
	for (std::vector<cv::Rect>::iterator itforerects = _foregroundrects.begin(); itforerects != _foregroundrects.end(); ++itforerects)
	{
		double t1 = (double)getTickCount();//��ȡϵͳ��ʱ��
		hog.detectMultiScale(Mat(image, *itforerects), _pedestrianrects, 0, cv::Size(8, 8), cv::Size(0, 0), 1.05, 1);//������ˣ�����Ĭ�ϲ���������
		size_t i, j;
		for (i = 0; i < _pedestrianrects.size(); ++i)//����regionsѰ����û�б�Ƕ�׵ĳ�����
		{
			Rect r = _pedestrianrects[i];
			for (j = 0; j < _pedestrianrects.size(); ++j)
				if (j != i && (r & _pedestrianrects[j]) == r)//���ʱǶ�׵ľ��˳�ѭ��
					break;
			if (j == _pedestrianrects.size())
				pedestrianrects_filtered.push_back(r);
		}
		//for (j = 0; j < pedestrianrects_filtered.size(); ++j)
		//{
		//	Rect r = pedestrianrects_filtered[j];
		//	//HOG��������صľ��ο�����ʵĿ���һЩ��������Ҫ�����ο���СһЩ�Եõ����õĽ��
		//	r.x += round(r.width*0.1);
		//	r.width = round(r.width*0.8);
		//	r.y += round(r.height*0.07);
		//	r.height = round(r.height*0.8);
		//	rectangle(image, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		//}
		_allpedestrianrects.push_back(pedestrianrects_filtered);//ѹջ����˴μ�⵽������
		t1 = (double)getTickCount() - t1;//��ü�����������ʱ�䲢���
		printf("������˵�ʱ��Ϊ = %gms/n", t1*1000. / ((double)getTickFrequency()));
		std::cout << std::endl;

		if (showpedestrianrect)
		{
			for (vector<Rect>::const_iterator itreg = pedestrianrects_filtered.begin(); itreg != pedestrianrects_filtered.end(); itreg++)//ѭ�������˴μ�⵽�����˵ľ��ο�
			{
				Rect r = *itreg;
				//HOG��������صľ��ο�����ʵĿ���һЩ��������Ҫ�����ο���СһЩ�Եõ����õĽ��
				r.x += round(r.width*0.1);
				r.width = round(r.width*0.8);
				r.y += round(r.height*0.07);
				r.height = round(r.height*0.8);
				rectangle(Mat(image, *itforerects), r, Scalar(0, 0, 255), 3, 8, 0);
				//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
			}
		}
		pedestrianrects_filtered.clear();
		_pedestrianrects.clear();
	}
	if (_allpedestrianrects.empty())
	{
		std::cout << "δ��⵽���ˣ��Ӷ��޷�����������⣡" << std::endl;
	}
}

int faceclass::facedect(cv::Mat& image, bool usegaussforegroundfordect)//���һ��ͼƬ�������
//��һ������Ϊ������ͼƬ���ڶ�������Ϊ�Ƿ���ǰ���˶����ڼ������������ֵ���������������(����������������)
//�����ɺ󣬼�⵽���������ο�洢��_allfacerects��
{
	_allfacerects.clear();//���¼��ʱ��Ҫ���ԭ���洢������
	_facerects.clear();

	_usegaussforegroundfordect = usegaussforegroundfordect;//���˲���ֵ���ݸ���ĳ�Ա����
	if (!_usegaussforegroundfordect)
	{
		_face_cascade.detectMultiScale(image, _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		if (_facerects.empty())
		{
			std::cout << "û�м�⵽������" << std::endl;
			return 0;
		}
		else
		{
			std::cout << "��⵽" << _facerects.size() << "��������" << std::endl;
			_allfacerects.push_back(_facerects);
			return _facerects.size();
		}
	}
	else
	{
		size_t facenum = 0;///��¼�ܹ���⵽����������

		for (int ROINo = 0; ROINo < _foregroundrects.size(); ROINo++)
		{
			double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
			//_frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			_face_cascade.detectMultiScale(cv::Mat(image, _foregroundrects[ROINo]), _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			facenum += _foregroundrects.size();
			//�����⵽�����ͱ���
			if (_foregroundrects.size()){
				_allfacerects.push_back(_facerects);//ѹջ����˴μ�⵽������
			}
			t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
			printf("���������ʱ��Ϊ = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
			cout << endl;

			for (int fNo = 0; fNo < int(_facerects.size()); fNo++)//ѭ�����������ľ��ο�
			{
				rectangle(image, Rect{ _foregroundrects[ROINo].x + _facerects[fNo].x, _foregroundrects[ROINo].y + _facerects[fNo].y, _facerects[fNo].width, _facerects[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
				//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
				//����Բ�������
				//Point center(rects[ROINo].x + faces[fNo].x + faces[fNo].width*0.5, rects[ROINo].y + faces[fNo].y + faces[fNo].height*0.5);//��Բ���ĵ�����Ҫ����ROI��������Ͻ������
				//ellipse(frame, center, Size(faces[fNo].width*0.5, faces[fNo].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 3, 8, 0);
			}
		}

		if (_allfacerects.size() == 0)
		{
			std::cout << "û�м�⵽������" << std::endl;
			return 0;
		}
		else
		{
			std::cout << "��⵽���������򱣴浽�˳�Ա����'_allfacerects'���棡\n����⵽" << facenum << "������" << std::endl;
			return facenum;
		}
	}
}

//�ȼ�����ˣ��ټ��������������
void faceclass::facedect(cv::Mat& image, int i)
{
	pedestriandect(image);//���ͼƬ�е�����

	for (std::vector<std::vector<cv::Rect>>::iterator itallpeddestrian = _allpedestrianrects.begin(); itallpeddestrian != _allpedestrianrects.end(); ++itallpeddestrian)
	{
		for (std::vector<cv::Rect>::iterator itpeddestrian = itallpeddestrian->begin(); itpeddestrian != itallpeddestrian->end(); ++itpeddestrian)
		{
			double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
			//_frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			_face_cascade.detectMultiScale(cv::Mat(image, *itpeddestrian), _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

			_allfacerects.push_back(_facerects);//ѹջ����˴μ�⵽������
			t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
			printf("���������ʱ��Ϊ = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
			cout << endl;

			for (int fNo = 0; fNo < int(_facerects.size()); fNo++)//ѭ�����������ľ��ο�
			{
				rectangle(image, Rect{ (*itpeddestrian).x + _facerects[fNo].x, (*itpeddestrian).y + _facerects[fNo].y, _facerects[fNo].width, _facerects[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
				//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
				//����Բ�������
				//Point center(rects[ROINo].x + faces[fNo].x + faces[fNo].width*0.5, rects[ROINo].y + faces[fNo].y + faces[fNo].height*0.5);//��Բ���ĵ�����Ҫ����ROI��������Ͻ������
				//ellipse(frame, center, Size(faces[fNo].width*0.5, faces[fNo].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 3, 8, 0);
			}
		}

		_allpedestrianfacerects.push_back(_allfacerects);
		if (_allpedestrianfacerects.size() == 0)
		{
			std::cout << "û�м�⵽������" << std::endl;
		}
		else
		{
			std::cout << "��⵽���������򱣴浽�˳�Ա������_allpedestrianfacerects�����棡" << std::endl;
		}
	}
}

////ֻ����Ϊ�Ҷ�ͼ�����򱨴�Ȼ�󽫻Ҷ�ͼ��һ��
//Mat& faceclass::toGrayscale(Mat& src)
//{
//	// ֻ����Ϊ��ͨ����
//	if (src.channels() != 1)
//	{
//		CV_Error(CV_StsBadArg, "ֻ֧�ֵ�ͨ���ľ���");
//	}
//	// ���������ع�һ�����ͼƬ
//	//Mat dst;
//	cv::normalize(src, src, 0, 255, NORM_MINMAX, CV_8UC1);
//	return src;
//}

/*�ṹ��˵��
struct _finddata_t
{
unsigned attrib;     //�ļ�����
time_t time_create;  //�ļ�����ʱ��
time_t time_access;  //�ļ���һ�η���ʱ��
time_t time_write;   //�ļ���һ���޸�ʱ��
_fsize_t size;  //�ļ��ֽ���
char name[_MAX_FNAME]; //�ļ���
};
//��FileName��������ƥ�䵱ǰĿ¼��һ���ļ�
_findfirst(_In_ const char * FileName, _Out_ struct _finddata64i32_t * _FindData);
//��FileName��������ƥ�䵱ǰĿ¼��һ���ļ�
_findnext(_In_ intptr_t _FindHandle, _Out_ struct _finddata64i32_t * _FindData);
//�ر�_findfirst���ص��ļ����
_findclose(_In_ intptr_t _FindHandle);*/

//�������ܣ�������·���µ����з�������������Բ���Ϊ��չ����ͼƬ��Ĭ��Ϊ".jpg"
bool faceclass::traversal(string fileextension)//filename������ͨ�����'��'����һ���ַ���'*'����0�������ַ�
{
	int jpgNum = 0;//��¼�ܹ��ҵ���ͼƬ�ĸ���
	char labelchar[10] = "\0";//�����ͼƬ����������ȡ���ı��
	int label = 0;
	_finddata_t fileInfo;//����ṹ��
	long handle = _findfirst(fileextension.c_str(), &fileInfo);//Ѱ�ҵ�һ��ͼƬ�ļ�

	if (handle == -1L)//���û���ҵ��ļ���������-1L
	{
		cerr << "δ�ҵ�ͼƬ�ļ���" << std::endl;
		return false;
	}

	do
	{
		if (fileInfo.name[0] != 'f')//���ͼƬ���Ʋ�����f��ͷ�򲻷���������������һ�β���
		{
			continue;//������break�������ֱ���˳�do-whileѭ��
		}
		for (unsigned int i = 0; i < strlen(fileInfo.name); i++)
		{
			if (fileInfo.name[i + 1] == '_')//�����»������˳�forѭ��
			{
				labelchar[i] = '\0';
				break;
			}
			labelchar[i] = fileInfo.name[i + 1];//ͼƬ���ֵĵ�һ����ĸΪf�����ñ���
		}
		label = atoi(labelchar);//��char�ͱ��ת����int��
		_labels.push_back(label);//�����ѹջ����������
		_faces.push_back(imread(fileInfo.name, 0));//���ҵ���ͼƬ�ԻҶ���ʽѹջ����faces�����򲻻��н��
		jpgNum++;//ͼƬ����1
		std::cout << fileInfo.name << std::endl;//����ҵ���ͼƬ������
	} while (_findnext(handle, &fileInfo) == 0);//Ѱ����һ��ͼƬ
	std::cout << " ��Ч .jpg ͼƬ�ļ��ĸ���Ϊ:  " << jpgNum << std::endl;//����ҵ���ͼƬ�ĸ���

	return true;
}

bool faceclass::setmodelno(int modelno)
//Ĭ�ϴ���LBP����ʶ��ģ�ͣ�����ͨ�������޸�
{
	_facemodelno = modelno;

	switch (_facemodelno)
	{
	case 1:
		_model = createEigenFaceRecognizer(10);//����PCAģ��
		std::cout << "����PCA����ʶ��ģ�ͣ�" << std::endl;
		return true;
		break;
	case 2:
		_model = createFisherFaceRecognizer();//����FisherFaceģ��
		std::cout << "����FisherFace����ʶ��ģ�ͣ�" << std::endl;
		return true;
		break;
	case 3:
		_model = createLBPHFaceRecognizer();//����LBPģ�ͣ���ģ�ͼ������Ч�����
		std::cout << "����LBP����ʶ��ģ�ͣ�" << std::endl;
		return true;
		break;
	default:
		std::cout << "����Ĳ���������Ҫ�󣬴�������ʶ��ģ��ʧ�ܣ�" << std::endl;
		return false;
		break;
	}
}

void faceclass::trainsavefacemodel()
{
	_model->train(_faces, _labels);//ѵ��ģ��
	switch (_facemodelno)
	{
	case 1:
		_model->save("PCAFace.xml");//��ѵ���õ�����PCAģ�ͱ����XML�ļ�
		std::cout << "����PCA����ʶ��ģ�ͣ�" << std::endl;
		break;
	case 2:
		_model->save("FisherFace.xml");//��ѵ���õ�����FisherFaceģ�ͱ����XML�ļ�
		std::cout << "����FisherFace����ʶ��ģ�ͣ�" << std::endl;
		break;
	case 3:
		_model->save("LBPHface.xml");//��ѵ���õ�����LBPģ�ͱ����XML�ļ�
		std::cout << "����LBP����ʶ��ģ�ͣ�" << std::endl;
		break;
	default:
		std::cout << "����Ĳ���������Ҫ�󣬱�������ʶ��ģ��ʧ�ܣ�" << std::endl;
		break;
	}
}

void faceclass::loadfacemodel()
{
	switch (_facemodelno)
	{
	case 1:
		_model->load("PCAFace.xml");//��ѵ���õ�����PCAģ�͵�XML�ļ����ص�ģ��
		std::cout << "����PCA����ʶ��ģ�ͣ�" << std::endl;
		break;
	case 2:
		_model->load("FisherFace.xml");//��ѵ���õ�����FisherFaceģ�͵�XML�ļ����ص�ģ��
		std::cout << "����FisherFace����ʶ��ģ�ͣ�" << std::endl;
		break;
	case 3:
		_model->load("LBPHface.xml");//��ѵ���õ�����LBPģ�͵�XML�ļ����ص�ģ��
		std::cout << "����LBP����ʶ��ģ�ͣ�" << std::endl;
		break;
	default:
		std::cout << "����Ĳ���������Ҫ�󣬼�������ʶ��ģ��ʧ�ܣ�" << std::endl;
		break;
	}
}

void faceclass::facecamshift(cv::Mat& image)
{
	cv::Rect trackWindow;//������ٵľ���
	//RotatedRect trackBox;//����һ����ת�ľ����������CamShift����
	int hsize = 16;//ÿһάֱ��ͼ�Ĵ�С
	float hranges[] = { 0, 180 };//hranges�ں���ļ���ֱ��ͼ������Ҫ�õ�
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//
	cv::Mat hsv, hue, mask, hist, backproj;

	//camshift��������
	for (std::vector<std::vector<cv::Rect>>::iterator itallf = _allfacerects.begin(); itallf != _allfacerects.end(); ++itallf)
	{
		for (std::vector<cv::Rect>::iterator itf = itallf->begin(); itf != itallf->end(); ++itf)
		{
			cv::cvtColor(image, hsv, CV_BGR2HSV);//��rgb����ͷ֡ת����hsv�ռ��
			//inRange�����Ĺ����Ǽ����������ÿ��Ԫ�ش�С�Ƿ���2��������ֵ֮�䣬�����ж�ͨ��,mask����0ͨ������Сֵ��Ҳ����h����
			//����������hsv��3��ͨ�����Ƚ�h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)�����3��ͨ�����ڶ�Ӧ�ķ�Χ�ڣ���
			//mask��Ӧ���Ǹ����ֵȫΪ1(0xff)������Ϊ0(0x00).
			cv::inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
			int ch[] = { 0, 0 };
			hue.create(hsv.size(), hsv.depth());//hue��ʼ��Ϊ��hsv��С���һ���ľ���ɫ���Ķ������ýǶȱ�ʾ�ģ�������֮�����120�ȣ���ɫ���180��
			cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);//��hsv��һ��ͨ��(Ҳ����ɫ��)�������Ƶ�hue�У�0��������

			//�˴��Ĺ��캯��roi�õ���Mat hue�ľ���ͷ����roi������ָ��ָ��hue����������ͬ�����ݣ�selectΪ�����Ȥ������
			trackWindow = *itf;
			cv::Mat roi(hue, trackWindow), maskroi(mask, trackWindow);//mask�����hsv����Сֵ

			//calcHist()������һ������Ϊ����������У���2��������ʾ����ľ�����Ŀ����3��������ʾ��������ֱ��ͼά��ͨ�����б���4��������ʾ��ѡ�����뺯��
			//��5��������ʾ���ֱ��ͼ����6��������ʾֱ��ͼ��ά������7������Ϊÿһάֱ��ͼ����Ĵ�С����8������Ϊÿһάֱ��ͼbin�ı߽�
			cv::calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);//��roi��0ͨ������ֱ��ͼ��ͨ��mask����hist�У�hsizeΪÿһάֱ��ͼ�Ĵ�С
			cv::normalize(hist, hist, 0, 255, CV_MINMAX);//��hist����������鷶Χ��һ��������һ����0~255

			cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);//����ֱ��ͼ�ķ���ͶӰ������hueͼ��0ͨ��ֱ��ͼhist�ķ���ͶӰ��������backproj��
			backproj &= mask;

			//opencv2.0�Ժ�İ汾��������ǰû��cv�����ˣ������������������2����˼�ĵ���Ƭ����ɵĻ�����ǰ���Ǹ�Ƭ�β����ɵ��ʣ����һ����ĸҪ
			//��д������Camshift�������һ����ĸ�Ǹ����ʣ���Сд������meanShift�����ǵڶ�����ĸһ��Ҫ��д
			_trackBox = cv::CamShift(backproj, trackWindow,               //trackWindowΪ���ѡ�������TermCriteriaΪȷ��������ֹ��׼��
				cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));//CV_TERMCRIT_EPS��ͨ��forest_accuracy,CV_TERMCRIT_ITER

			if (trackWindow.area() <= 1)                                                  //��ͨ��max_num_of_trees_in_the_forest  
			{
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
					trackWindow.x + r, trackWindow.y + r) &
					Rect(0, 0, cols, rows);//Rect����Ϊ�����ƫ�ƺʹ�С������һ��������Ϊ��������Ͻǵ����꣬�����ĸ�����Ϊ����Ŀ�͸�
			}

			cv::ellipse(image, _trackBox, Scalar(255, 255, 0), 3, CV_AA);//���ٵ�ʱ������ԲΪ����Ŀ��
		}
	}
}

//Ĭ�ϴ�����ͷ����Ҫ���ļ���Ҫ����һ��������Ϊ-2�������ļ������ݸ��ڶ�������
bool faceclass::opencamera(int cameranum, string filename)
{
	if (cameranum == -2){
		_capture.open(filename);
	}
	else{
		_capture.open(0);
	}
	if (!_capture.isOpened())
	{
		std::cout << "������ͷʧ�ܣ�" << std::endl;
		return false;
	}
	else
	{
		std::cout << "����ͷ�Ѵ򿪣�" << std::endl;
		return true;
	}
}

//������p����Ԥ��������������q���˳�����Ԥ��
void faceclass::predect(bool usepedestrianrects, bool savevideobool)
{
	map<int, string> id_dict;
	id_dict.insert(map<int, string>::value_type(1, "������"));
	id_dict.insert(map<int, string>::value_type(2, "���"));
	id_dict.insert(map<int, string>::value_type(3, "���շ�"));
	id_dict.insert(map<int, string>::value_type(4, "����"));
	id_dict.insert(map<int, string>::value_type(5, "�̻���"));

	if (opencamera())//����ɹ�������ͷ
	{
		if (usepedestrianrects)
		{
			guassforeground(_frame, 0.05, true);
			getforegroundrect();
			pedestriandect(_frame);
		}

		if (savevideobool)
		{
			savevideoinit();
		}

		cv::Mat temp;//������Ƶ֡
		cv::Mat face;//��������

		bool ispre = false;
		int predictedLabel = 0;
		int facenum = 0;
		while (_capture.read(_frame))
		{
			char c = waitKey(33);//����֡��
			cv::imshow("predect", _frame);//��ʾ��Ƶ

			if (c == 'p')//����p��Ԥ������
			{
				ispre = true;
				std::cout << std::endl << "Ԥ���֡��Ƶ������" << std::endl;
			}

			if (c == 'q')//����q�˳�����Ԥ��
			{
				std::cout << "ֹͣԤ������" << std::endl;
				break;
			}

			if (ispre)//�����û���������ĸp������ʼ��Ⲣʶ������
			{
				ispre = false;//����Ԥ���־λΪ�٣�ȷ��ÿ��ֻԤ��һ������

				_frame.copyTo(temp);//���Ƶ�ǰ��Ƶ֡				

				if (usepedestrianrects)//���ʹ�����˼��ľ��ο����������
				{
					facedect(temp,1);//�ڵ�ǰ֡�м������
					for (std::vector<std::vector<std::vector<cv::Rect>>>::iterator itallpedfacerects = _allpedestrianfacerects.begin(); itallpedfacerects != _allpedestrianfacerects.end(); ++itallpedfacerects)
					{
						for (std::vector<std::vector<cv::Rect>>::iterator itpedfacerects = itallpedfacerects->begin(); itpedfacerects != itallpedfacerects->end(); ++itpedfacerects)
						{
							for (std::vector<cv::Rect>::iterator itrects = itpedfacerects->begin(); itrects != itpedfacerects->end(); ++itrects)
							{
								face = cv::Mat(_frame, *itrects);
								resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��ȷ����ģ��ѵ��ʱ��ȡ��ͼƬ��С��ͬ
								cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
								//toGrayscale(face);//��һ��
								cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);
								predictedLabel = _model->predict(face);
								++facenum;
								cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//����⵽��������ʾ����
								std::cout << "��⵽��" << id_dict[predictedLabel] << std::endl;
							}
						}
					}
				}
				else//
				{
					int facenum = facedect(temp);//�ڵ�ǰ֡�м������
					for (std::vector<std::vector<cv::Rect>>::iterator itallrects = _allfacerects.begin(); itallrects != _allfacerects.end(); ++itallrects)
					{
						for (std::vector<cv::Rect>::iterator itrects = itallrects->begin(); itrects != itallrects->end(); ++itrects)
						{
							face = cv::Mat(_frame, *itrects);
							resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��ȷ����ģ��ѵ��ʱ��ȡ��ͼƬ��С��ͬ
							cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
							//toGrayscale(face);//��һ��
							cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);
							predictedLabel = _model->predict(face);
							++facenum;
							cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//����⵽��������ʾ����
							std::cout << "��⵽��" << id_dict[predictedLabel] << std::endl;
						}
					}
				}
			}
			if (savevideobool)
			{
				_capsave << _frame;//������Ƶ֡
			}
		}
	}
}



bool faceclass::savevideoinit()//��ʼ��������Ƶ����
{
	_capture >> _frame;//��ȡһ֡����Ƶ�ļ�
	_capsave.open("���.avi", CV_FOURCC('M', 'J', 'P', 'G'), 33, _frame.size(), 1);
	_capsave << _frame;//����һ֡����Ƶ�ļ�
	if (!_capsave.isOpened())//�жϱ�����Ƶ�Ƿ���ȷ��ʼ��
	{
		cout << "������Ƶʧ��!" << endl;
		return false;
	}
	else
	{
		std::cout << "������Ƶ������ʼ���ɹ���" << std::endl;
		return true;
	}
}


