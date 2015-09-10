
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
#include <fstream>
#include "faceclass.h"

//extern void runvedio(string filename, int& func);
//extern void train_new_model(string fileextension = "*.jpg");
//extern void smartdect(int& func, bool dect_face = false, bool dect_pedestrian = true, bool save_videobool = false, bool showforeground = false, bool use_camshift = true);//����Ϊ����ģ�ͼ���ģ�͵����
//extern void userdect(int& func, bool dect_face = true, bool dect_pedestrian = true, bool save_videobool = false, bool use_camshift = true);

using namespace cv;
using namespace std;

faceclass::faceclass() {
	_vedio_open = false;
	_facemodelno = 3;
	_func = 1;
}
faceclass::~faceclass() {}

//faceclass.hͷ�ļ�����cascade_name��Ĭ��ֵ
bool faceclass::addcascade(char* cascade_name) {
	_cascade_name = cascade_name;

	if (!_face_cascade.load(_cascade_name)) {//�ж�Haar���������Ƿ�ɹ�

		std::cout << "�޷����ؼ����������ļ���" << std::endl;
		return false;
	}
	else {
		std::cout << "�ɹ����ؼ���������" << cascade_name << std::endl;
		return true;
	}
}

void faceclass::guassforeground(cv::Mat& image, double learningspeed, bool showforeground) {
	//�ڶ��������������ĸ������ʣ��������������������Ƿ���ʾ������ǰ��

	// �˶�ǰ����⣬�����±���;_mog�Ƕ�����faceclass�е�һ����ȡ������BackgroundSubtractorMOG2�Ķ���
	_mog(image, _foreground, learningspeed);//learningspeedΪ�������ʣ�Ĭ��Ϊ0.05�����Լ�����
	//ȥ������
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);//����
	erode(_foreground, _foreground, Mat(), Point(-1, -1), 2);//��ʴ
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);

	//_mog.getBackgroundImage(_background);   // ���ص�ǰ����ͼ��

	_showforeground = showforeground;
}

int faceclass::getforegroundrect(bool showforegroundrect) { //�������������Ƿ���ʾǰ���˶�������ο�ֻ����ǰ����ʾ��ͬʱ��Ч
	//ÿ�����¼�ⶼҪ�����һ�ε�ǰ����
	_foregroundrects.clear();
	cv::Mat fgdrect;
	_foreground.copyTo(fgdrect);//����ǰ�������ڼ������
	fgdrect = fgdrect > 50;//�������ش���50�����ػ���Ϊ255����������Ϊ0
	//���ǰ��������
	findContours(fgdrect, _contours, _hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//��������������ع����

	//_foregroundrects�洢������������Ӿ��Σ���boundingRect��������
	int fore_rect_num = 0;
	int idx = 0;//��������ѭ��
	if (_contours.size()) {//������ϴ��жϣ�������Ƶ��ֻ�б���ʱ�����
		for (; idx >= 0; idx = _hierarchy[idx][0]) {//�ҵ��������������hierarchy[idx][0]��ָ����һ����������û����һ��������hierarchy[idx][0]Ϊ������
			fore_rect_num++;
			if (fabs(contourArea(Mat(_contours[idx]))) > 5000) { //�����ǰ������������ڴ�ǰ���������ֵ���򱣴浱ǰֵ	
				
				cv::Rect fore_rect = boundingRect(_contours[idx]);

				if (fore_rect.x < 0){
					fore_rect.x = 0;
				}
				else if (fore_rect.x > fgdrect.size().width){
					fore_rect.x = fgdrect.size().width;
				}

				if (fore_rect.y < 0){
					fore_rect.y = 0;
				}
				else if (fore_rect.y > fgdrect.size().height){
					fore_rect.y = fgdrect.size().height;
				}

				if (fore_rect.x + fore_rect.width > fgdrect.size().width){
					fore_rect.width = fgdrect.size().width - fore_rect.x;
				}

				if (fore_rect.y + fore_rect.height > fgdrect.size().height){
					fore_rect.height = fgdrect.size().height - fore_rect.y;
				}

				_foregroundrects.push_back(fore_rect);//ѹջ���������������Ӿ���
			}
			if (fore_rect_num > 5){
				//break;
			}
		}
	}

	if (_showforeground && showforegroundrect) {//�����ʾǰ�������������Ƿ���ʾǰ���˶��ľ��ο�
	
		for (vector<Rect>::iterator it = _foregroundrects.begin(); it != _foregroundrects.end(); it++) {//�������з�����������Ӿ���
			rectangle(_foreground, *it, Scalar(255, 255, 255), 3, 8, 0);//��ǰ������������������Ӿ��ο�
		}
	}

	if (_showforeground && (!_foreground.empty())) {
		cv::imshow("foreground", _foreground);//��ʾǰ��
		//imshow("background", _background);//��ʾ����
	}

	return _foregroundrects.size();
}

int faceclass::pedestriandect(cv::Mat& image, bool usegaussforegroundfordect) {
	_allpedestrianrects.clear();//���¼����Ҫ���
	HOGDescriptor hog;//����HOG���󣬲���Ĭ�ϲ���
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//����Ĭ�ϵ����˼�������
	vector<Rect> pedestrianrects_filtered;//�洢������˵ı߽�
	size_t pedestrian_no = 0;
	if (usegaussforegroundfordect) {
		for (std::vector<cv::Rect>::iterator itforerects = _foregroundrects.begin(); itforerects != _foregroundrects.end(); ++itforerects) {
			double t1 = (double)getTickCount();//��ȡϵͳ��ʱ��
			hog.detectMultiScale(Mat(image, *itforerects), _pedestrianrects, 0, cv::Size(8, 8), cv::Size(0, 0), 1.05, 1);//������ˣ�����Ĭ�ϲ���������
			size_t i, j;
			for (i = 0; i < _pedestrianrects.size(); ++i) { //����regionsѰ����û�б�Ƕ�׵ĳ�����
				Rect r = _pedestrianrects[i];
				for (j = 0; j < _pedestrianrects.size(); ++j)
					if (j != i && (r & _pedestrianrects[j]) == r)//���ʱǶ�׵ľ��˳�ѭ��
						break;
				if (j == _pedestrianrects.size())
					pedestrianrects_filtered.push_back(r);
			}
			for (j = 0; j < pedestrianrects_filtered.size(); ++j) {
				Rect r = pedestrianrects_filtered[j];
				//HOG��������صľ��ο�����ʵĿ���һЩ��������Ҫ�����ο���СһЩ�Եõ����õĽ��
				r.x += round(r.width * 0.07);
				if (r.x > itforerects->size().width) {
					r.x = itforerects->size().width;
				}
				r.width = round(r.width * 0.8);
				if (r.x + r.width > itforerects->size().width) {
					r.width = itforerects->size().width - r.x;
				}
				r.y += round(r.height * 0.07);
				if (r.y > itforerects->size().height) {
					r.y = itforerects->size().height;
				}
				r.height = round(r.height * 0.8);
				if (r.y + r.height > itforerects->size().height) {
					r.height = itforerects->size().height;
				}
				rectangle(image, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
			}
			_allpedestrianrects.push_back(pedestrianrects_filtered);//ѹջ����˴μ�⵽������
			t1 = (double)getTickCount() - t1;//��ü�����������ʱ�䲢���
			pedestrian_no += pedestrianrects_filtered.size();//�ۼӵõ����е�������Ŀ
			std::printf("������˵�ʱ��Ϊ = %gms/n", t1 * 1000. / ((double)getTickFrequency()));
			std::cout << std::endl;

			//for (vector<Rect>::const_iterator itreg = pedestrianrects_filtered.begin(); itreg != pedestrianrects_filtered.end(); itreg++)//ѭ�������˴μ�⵽�����˵ľ��ο�
			//{
			//	Rect r = *itreg;
			//	//HOG��������صľ��ο�����ʵĿ���һЩ��������Ҫ�����ο���СһЩ�Եõ����õĽ��
			//	r.x += round(r.width*0.1);
			//	r.width = round(r.width*0.8);
			//	r.y += round(r.height*0.07);
			//	r.height = round(r.height*0.8);
			//	rectangle(Mat(image, *itforerects), r, Scalar(0, 0, 255), 3, 8, 0);
			//	//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
			//}

			pedestrianrects_filtered.clear();
			_pedestrianrects.clear();
		}
		return pedestrian_no;
	}
	else {
		double t1 = (double)getTickCount();//��ȡϵͳ��ʱ��
		hog.detectMultiScale(image, _pedestrianrects, 0, cv::Size(8, 8), cv::Size(0, 0), 1.05, 1);//������ˣ�����Ĭ�ϲ���������
		size_t i, j;
		for (i = 0; i < _pedestrianrects.size(); ++i) { //����regionsѰ����û�б�Ƕ�׵ĳ�����
			Rect r = _pedestrianrects[i];
			for (j = 0; j < _pedestrianrects.size(); ++j)
				if (j != i && (r & _pedestrianrects[j]) == r)//���ʱǶ�׵ľ��˳�ѭ��
					break;
			if (j == _pedestrianrects.size())
				pedestrianrects_filtered.push_back(r);
		}
		for (j = 0; j < pedestrianrects_filtered.size(); ++j) {
			Rect r = pedestrianrects_filtered[j];
			//HOG��������صľ��ο�����ʵĿ���һЩ��������Ҫ�����ο���СһЩ�Եõ����õĽ��
			r.x += round(r.width * 0.07);
			if (r.x > image.size().width) {
				r.x = image.size().width;
			}
			r.width = round(r.width * 0.8);
			if (r.x + r.width > image.size().width) {
				r.width = image.size().width - r.x;
			}
			r.y += round(r.height * 0.07);
			if (r.y > image.size().height) {
				r.y = image.size().height;
			}
			r.height = round(r.height * 0.8);
			if (r.y + r.height > image.size().height) {
				r.height = image.size().height;
			}
			rectangle(image, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}
		_allpedestrianrects.push_back(pedestrianrects_filtered);//ѹջ����˴μ�⵽������
		t1 = (double)getTickCount() - t1;//��ü�����������ʱ�䲢���
		std::printf("������˵�ʱ��Ϊ = %gms/n", t1 * 1000. / ((double)getTickFrequency()));
		std::cout << std::endl;

		pedestrian_no = pedestrianrects_filtered.size();//��⵽�����˸���
		pedestrianrects_filtered.clear();
		_pedestrianrects.clear();

		return pedestrian_no;
	}

	if (_allpedestrianrects.empty()) {
		std::cout << "δ��⵽���ˣ��Ӷ��޷�����������⣡" << std::endl;
		return 0;
	}
}

int faceclass::facedect(cv::Mat& image, bool usegaussforegroundfordect) {    //���һ��ͼƬ�������
//��һ������Ϊ������ͼƬ���ڶ�������Ϊ�Ƿ���ǰ���˶����ڼ������������ֵ���������������(����������������)
//�����ɺ󣬼�⵽���������ο�洢��_allfacerects��

	_allfacerects.clear();//���¼��ʱ��Ҫ���ԭ���洢������
	_facerects.clear();
	_allface_id.clear();
	_usegaussforegroundfordect = usegaussforegroundfordect;//���˲���ֵ���ݸ���ĳ�Ա����
	//�����ʹ��ǰ����Ϣ
	if (!_usegaussforegroundfordect) {
		_face_cascade.detectMultiScale(image, _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		if (_facerects.empty()) {
			std::cout << "û�м�⵽������" << std::endl;
			return 0;
		}
		else {
			for (int fNo = 0; fNo < int(_facerects.size()); fNo++) {   //ѭ�����������ľ��ο�
				cv::rectangle(image, _facerects[fNo], Scalar(255, 0, 255), 3, 8, 0);
				//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
			}
			std::cout << "��⵽" << _facerects.size() << "��������" << std::endl;
			_allfacerects.push_back(_facerects);
			return _facerects.size();
		}
	}
	else { //���ʹ��ǰ����Ϣ

		size_t facenum = 0;//��¼�ܹ���⵽����������

		for (int ROINo = 0; ROINo < _foregroundrects.size(); ROINo++) {
			double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
			//_frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			_face_cascade.detectMultiScale(cv::Mat(image, _foregroundrects[ROINo]), _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			facenum += _facerects.size();
			//�����⵽�����ͱ���
			if (_foregroundrects.size()) {
				_allfacerects.push_back(_facerects);//ѹջ����˴μ�⵽������
			}
			t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
			printf("���������ʱ��Ϊ = %gms/n", t / ((double)cvGetTickFrequency() * 1000.));
			cout << endl;

			for (int fNo = 0; fNo < int(_facerects.size()); fNo++) { //ѭ�����������ľ��ο�
				cv::rectangle(image, Rect{ _foregroundrects[ROINo].x + _facerects[fNo].x, _foregroundrects[ROINo].y + _facerects[fNo].y, _facerects[fNo].width, _facerects[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
				//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
			}
		}

		if (_allfacerects.size() == 0) {
			std::cout << "û�м�⵽������" << std::endl;
			return 0;
		}
		else {
			std::cout << "��⵽���������򱣴浽�˳�Ա����'_allfacerects'���棡\n����⵽" << facenum << "������" << std::endl;
			return facenum;
		}
	}
}


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

	do {
		if (fileInfo.name[0] != 'f')//���ͼƬ���Ʋ�����f��ͷ�򲻷���������������һ�β���
		{
			continue;//������break�������ֱ���˳�do-whileѭ��
		}
		for (unsigned int i = 0; i < strlen(fileInfo.name); i++) {
			if (fileInfo.name[i + 1] == '_')//�����»������˳�forѭ��
			{
				labelchar[i] = '\0';
				break;
			}
			labelchar[i] = fileInfo.name[i + 1];//ͼƬ���ֵĵ�һ����ĸΪf�����ñ���
		}
		label = atoi(labelchar);//��char�ͱ��ת����int��
		_labels.push_back(label);//�����ѹջ����������
		cv::Mat face_picture = imread(fileInfo.name, 0);
		resize(face_picture, face_picture, cv::Size{ 200, 200 });
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
		_model = createEigenFaceRecognizer(10);//����ΪPCAģ��
		std::cout << "����ΪPCA����ʶ��ģ�ͣ�" << std::endl;
		return true;
		break;
	case 2:
		_model = createFisherFaceRecognizer();//����ΪFisherFaceģ��
		std::cout << "����ΪFisherFace����ʶ��ģ�ͣ�" << std::endl;
		return true;
		break;
	case 3:
		_model = createLBPHFaceRecognizer(1,8,8,8,123.0);//����ΪLBPģ�ͣ���ģ�ͼ������Ч�����
		//_model->set("threshold", 80);
		std::cout << "����ΪLBP����ʶ��ģ�ͣ�" << std::endl;
		return true;
		break;
	default:
		std::cout << "����Ĳ���������Ҫ����������ʶ��ģ��ʧ�ܣ�" << std::endl;
		return false;
		break;
	}
}

void faceclass::trainsavefacemodel()
{
	_model->train(_faces, _labels);//ѵ��ģ��
	switch (_facemodelno) {
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

void faceclass::train_new_model(string fileextension) {
	traversal(fileextension);
	trainsavefacemodel();
}

void faceclass::loadfacemodel() {
	switch (_facemodelno) {
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

void faceclass::facecamshift(cv::Mat& image, int face_id) {
	cv::Rect trackWindow;//������ٵľ���
	//RotatedRect trackBox;//����һ����ת�ľ����������CamShift����
	int hsize = 16;//ÿһάֱ��ͼ�Ĵ�С
	float hranges[] = { 0, 180 };//hranges�ں���ļ���ֱ��ͼ������Ҫ�õ�
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//
	cv::Mat hsv, hue, mask, hist, backproj;
	int facenumforid = 0;
	//camshift��������
	for (std::vector<std::vector<cv::Rect>>::iterator itallf = _allfacerects.begin(); itallf != _allfacerects.end(); ++itallf) {
		for (std::vector<cv::Rect>::iterator itf = itallf->begin(); itf != itallf->end(); ++itf) {
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

			if (trackWindow.area() <= 1) {                                                 //��ͨ��max_num_of_trees_in_the_forest
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
				                   trackWindow.x + r, trackWindow.y + r) &
				              Rect(0, 0, cols, rows);//Rect����Ϊ�����ƫ�ƺʹ�С������һ��������Ϊ��������Ͻǵ����꣬�����ĸ�����Ϊ����Ŀ�͸�
			}

			//cv::ellipse(image, _trackBox, Scalar(255, 255, 0), 3, CV_AA);//���ٵ�ʱ������ԲΪ����Ŀ��
			Rect facerecttodraw;
			facerecttodraw.x = _trackBox.center.x - _trackBox.size.width / 2;
			facerecttodraw.y = _trackBox.center.y - _trackBox.size.height / 2;
			facerecttodraw.width = _trackBox.size.width;
			facerecttodraw.height = _trackBox.size.height;
			putText(image, _id_dict[_allface_id[facenumforid]], cv::Point(facerecttodraw.x, facerecttodraw.y), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));//��ͼƬ������ַ�
			cv::rectangle(image, facerecttodraw, Scalar(255, 255, 0), 3, 0);

			facenumforid++;

		}
	}
}

//Ĭ�ϴ�����ͷ����Ҫ���ļ���Ҫ����һ��������Ϊ-2�������ļ������ݸ��ڶ�������
bool faceclass::opencamera(string filename, int cameranum) {
	//if (cameranum == -2 && filename != "nothing"){
	if (filename != "nothing") {
		_capture.open(filename);
	}
	else if (cameranum != -2) {
		_capture.open(cameranum);
	}

	if (!_capture.isOpened()) {
		std::cout << "����Ƶ�ļ�ʧ�ܣ�" << std::endl;
		return false;
	}
	else {
		std::cout << "��Ƶ�ļ��Ѵ򿪣�" << std::endl;
		return true;
	}
}

void faceclass::runvedio(string filename, int& func) {
	//if (opencamera(filename)){
	//	_vedio_open = true;

	//while (_capture.read(_frame)){
	//	char c = waitKey(33);//����֡��
	//	cv::imshow("vedio", _frame);//��ʾ��Ƶ
	//}
	//}

	if (func == 1) {
		while (_capture.read(_frame) && func == 1) {
			char c = waitKey(33);//����֡��

			if (c == 'q') {
				break;
			}

			cv::imshow("vedio", _frame);//��ʾ��Ƶ

			//if (_func != 1){
			//	break;
			//}
		}
	}
}

//�洢��ǩ�������ֵĶ�Ӧ��ϵ
void faceclass::insertdict(int lablenum, string name) {
	_id_dict.insert(pair<int, string>(lablenum, name));
}

//������p����Ԥ��������������q���˳�����Ԥ��
void faceclass::smartdect(int& func, bool dect_face, bool dect_pedestrian, bool save_videobool, bool showforeground, bool use_camshift) {
	//_id_dict.insert(map<int, string>::value_type(1, "kyle"));
	//_id_dict.insert(map<int, string>::value_type(2, "lijuan"));
	//_id_dict.insert(map<int, string>::value_type(3, "fanshu"));
	//_id_dict.insert(map<int, string>::value_type(4, "yangzai"));
	//_id_dict.insert(map<int, string>::value_type(5, "xiaocai"));
	//_id_dict.insert(pair<int, string>(-1, "unknow"));


	_id_dict.insert(map<int, string>::value_type(1, "kyle"));
	_id_dict.insert(map<int, string>::value_type(2, "qianli"));
	//_id_dict.insert(map<int, string>::value_type(3, "fanshu"));
	//_id_dict.insert(map<int, string>::value_type(4, "yangzai"));
	//_id_dict.insert(map<int, string>::value_type(5, "xiaocai"));
	_id_dict.insert(pair<int, string>(-1, "unknow"));

	//if (opencamera()){//����ɹ�������ͷ
	//if (!_frame.empty()){//����ɹ�������ͷ
	if (func == 2) {

		//if (savevideobool){
		//	savevideoinit();
		//}

		size_t frame_no = 0;
		cv::Mat face;//��������
		fstream outfile;
		char timetmp[20];
		bool ispre = false;
		string record = "record.txt";
		int predictedLabel = 0;
		int facenum = 0;
		while (_capture.read(_frame) && func == 2) {
			char c = waitKey(33);//����֡��

			if (c == 'q') {
				break;
			}

			if (frame_no % 30 == 0 ) {
				guassforeground(_frame, 0.05, showforeground);
				int forenum = getforegroundrect();
				if (forenum) {
					if (dect_pedestrian) { //����������
						pedestriandect(_frame, true);//�ڵ�ǰ֡�м�����ˣ��˺�������Զʹ��ǰ����Ϣ���ڶ�������Ϊtrue��
					}
					if (dect_face) {
						int facenum = facedect(_frame, true);//�ڵ�ǰ֡�м���������˺�������Զʹ��ǰ����Ϣ���ڶ�������Ϊtrue��
						for (std::vector<std::vector<cv::Rect>>::iterator itallrects = _allfacerects.begin(); itallrects != _allfacerects.end(); ++itallrects) {
							for (std::vector<cv::Rect>::iterator itrects = itallrects->begin(); itrects != itallrects->end(); ++itrects) {
								face = cv::Mat(_frame, *itrects);
								resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��ȷ����ģ��ѵ��ʱ��ȡ��ͼƬ��С��ͬ
								cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
								cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);
								predictedLabel = _model->predict(face);//����ģ��ʶ������ID
								_id = predictedLabel;
								_allface_id.push_back(_id);
								//cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//����⵽��������ʾ����
								std::cout << "��⵽��" << _id_dict[predictedLabel] << std::endl;
								string face_id = _id_dict[predictedLabel];
								putText(_frame, face_id, cv::Point(itrects->x, itrects->y), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));//��ͼƬ������ַ�

								time_t qiandaotime = time(0);//��ȡϵͳ��ǰ��ʱ��
								strftime(timetmp, 20, "%Y/%m/%d %H:%M:%S", localtime(&qiandaotime));
								//����¼���浽record.txt�ļ���
								outfile.open(record, ios::app);//��׷�ӵķ�ʽд�ļ�
								outfile << "����ʱ�� " << timetmp << " " << _id_dict[predictedLabel] << "�����ڼ������" << endl;//��ǩ�����˱�ź�ʱ���¼����
							}
						}
					}
				}
			}


			if (use_camshift) {
				facecamshift(_frame, _id);
			}

			if (dect_pedestrian){
				for (vector<vector<cv::Rect>>::iterator it = _allpedestrianrects.begin(); it != _allpedestrianrects.end(); ++it){
					for (vector<cv::Rect>::iterator it_inner = it->begin(); it_inner != it->end(); ++it_inner){
						rectangle(_frame, it_inner->tl(), it_inner->br(), cv::Scalar(0, 255, 0), 3);
					}
				}
			}

			if (save_videobool) {
				_capsave << _frame;//������Ƶ֡
			}

			cv::imshow("predect", _frame);//��ʾ��Ƶ

			frame_no++;

			//if (_func != 2){
			//	break;
			//}
		}
	}
}

//�����ʹ�ñ�����ⷽ����ÿ10֡��Ƶ���һ��
void faceclass::userdect(int& func, bool dect_face, bool dect_pedestrian, bool save_videobool, bool use_camshift) {
	//_id_dict.insert(map<int, string>::value_type(1, "kyle"));
	//_id_dict.insert(map<int, string>::value_type(2, "lijuan"));
	//_id_dict.insert(map<int, string>::value_type(3, "fanshu"));
	//_id_dict.insert(map<int, string>::value_type(4, "yangzai"));
	//_id_dict.insert(map<int, string>::value_type(5, "xiaocai"));
	//_id_dict.insert(pair<int, string>(-1, "unknow"));

	_id_dict.insert(map<int, string>::value_type(1, "kyle"));
	_id_dict.insert(map<int, string>::value_type(2, "qianli"));
	//_id_dict.insert(map<int, string>::value_type(3, "fanshu"));
	//_id_dict.insert(map<int, string>::value_type(4, "yangzai"));
	//_id_dict.insert(map<int, string>::value_type(5, "xiaocai"));
	_id_dict.insert(pair<int, string>(-1, "unknow"));

	//if (opencamera()){//����ɹ�������ͷ
	//if (!_frame.empty()){//����ɹ�������ͷ
	if (func == 3) {

		//if (savevideobool){
		//	savevideoinit();
		//}

		size_t frame_no = 0;
		cv::Mat face;//��������
		fstream outfile;
		char timetmp[20];
		bool ispre = false;
		string record = "record.txt";
		int predictedLabel = 0;
		int facenum = 0;
		while (_capture.read(_frame) && func == 3) {
			char c = waitKey(33);//����֡��

			if (c == 'q') {
				break;
			}

			if (dect_pedestrian && frame_no % 5 == 0) { //����������
				pedestriandect(_frame, false);//�ڵ�ǰ֡�м�����ˣ��˺�������Զ��ʹ��ǰ����Ϣ���ڶ�������Ϊfalse��
			}
			if (dect_face && frame_no % 30 == 0) {
				int facenum = facedect(_frame, false);//�ڵ�ǰ֡�м���������˺�������Զ��ʹ��ǰ����Ϣ���ڶ�������Ϊfalse��
				for (std::vector<std::vector<cv::Rect>>::iterator itallrects = _allfacerects.begin(); itallrects != _allfacerects.end(); ++itallrects) {
					for (std::vector<cv::Rect>::iterator itrects = itallrects->begin(); itrects != itallrects->end(); ++itrects) {
						face = cv::Mat(_frame, *itrects);
						resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��ȷ����ģ��ѵ��ʱ��ȡ��ͼƬ��С��ͬ
						cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
						cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);
						predictedLabel = _model->predict(face);
						_id = predictedLabel;
						_allface_id.push_back(_id);

						//cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//����⵽��������ʾ����
						std::cout << "��⵽��" << _id_dict[predictedLabel] << std::endl;
						string face_id = _id_dict[predictedLabel];
						putText(_frame, face_id, cv::Point(itrects->x, itrects->y), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));//��ͼƬ������ַ�
						time_t qiandaotime = time(0);//��ȡϵͳ��ǰ��ʱ��
						strftime(timetmp, 20, "%Y/%m/%d %H:%M:%S", localtime(&qiandaotime));
						//����¼���浽record.txt�ļ���
						outfile.open(record, ios::app);//��׷�ӵķ�ʽд�ļ�
						outfile << "����ʱ�䣺 " << timetmp << " " << _id_dict[predictedLabel] << "�����ڼ������" << endl;//����⵽�ĵ����˱�ź�ʱ���¼����
					}
				}
			}

			if (use_camshift) {
				facecamshift(_frame, _id);
			}

			if (dect_pedestrian){
				for (vector<vector<cv::Rect>>::iterator it = _allpedestrianrects.begin(); it != _allpedestrianrects.end(); ++it){
					for (vector<cv::Rect>::iterator it_inner = it->begin(); it_inner != it->end(); ++it_inner){
						cv::rectangle(_frame, it_inner->tl(), it_inner->br(), cv::Scalar(0, 255, 0), 3);
					}
				}
			}

			if (save_videobool) {
				_capsave << _frame;//������Ƶ֡
			}

			cv::imshow("predect", _frame);//��ʾ��Ƶ

			frame_no++;

			//if (_func != 3){
			//	break;
			//}
		}
	}
}


bool faceclass::savevideoinit() { //��ʼ��������Ƶ����
	_capture >> _frame;//��ȡһ֡����Ƶ�ļ�
	_capsave.open("���.avi", CV_FOURCC('M', 'J', 'P', 'G'), 33, _frame.size(), 1);
	_capsave << _frame;//����һ֡����Ƶ�ļ�
	if (!_capsave.isOpened()) { //�жϱ�����Ƶ�Ƿ���ȷ��ʼ��
		cout << "������Ƶʧ��!" << endl;
		return false;
	}
	else {
		std::cout << "������Ƶ������ʼ���ɹ���" << std::endl;
		return true;
	}
}


