#ifndef FACECLASS_H_
#define FACECLASS_H_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/video/background_segm.hpp"
#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//�����ļ�ͼƬ��ʱ����Ҫ��ͷ�ļ�
//#include<conio.h>

using namespace cv;
using namespace std;

class faceclass
{
public:

	explicit faceclass();
	~faceclass();
	bool addcascade(char* cascade_name = "haarcascade_frontalface_alt.xml");//����Haar��������������
	void guassforeground(cv::Mat& image, double learningspeed = 0.05, bool showforeground = false);
	void getforegroundrect(bool showforegroudrect = true);

	void pedestriandect(cv::Mat& image, bool showpedestrianrect = true);//�������

	int facedect(cv::Mat& image, bool usegaussforegroundfordect = false);//���һ��ͼƬ�������
	void facedect(cv::Mat& image, int i);//�ڶ�����������Ϊ�˺�������

	cv::Mat& toGrayscale(cv::Mat& src);
	//cv::Mat opencamera();//������ͷ������һ֡��Ƶ
	bool cheakinputisnum(string inputlable);//�������ı���Ƿ�����Ч������
	string imagenamegen(int facelable, int facenum);
	void takphoto();//�ɼ�����

	bool traversal(string fileextension = "*.jpg");//������·���µ����з�������������Բ���Ϊ��չ����ͼƬ
	void trainsavefacemodel();
	void loadfacemodel();

	void facecamshift(cv::Mat& image);

	bool setmodelno(int modelno = 3);
	bool opencamera();
	void predect(bool usepedestrianrects = false, bool savevideobool = false);//����Ϊ����ģ�ͼ���ģ�͵����
	void showeigenface(bool eigenface = true);
	bool savevideoinit();

private:
	std::vector<cv::Mat> _faces;// ����ͼƬ����������Ӧ��ŵ�����
	std::vector<int> _labels;

	cv::CascadeClassifier _face_cascade;
	char* _cascade_name;

	cv::Mat _frame;
	cv::Mat _image;

	cv::Mat _foreground;//����ǰ��֡����
	cv::Mat _background;//���汳��֡����
	cv::BackgroundSubtractorMOG2 _mog;
	bool _showforeground;
	std::vector<std::vector<cv::Point> > _contours;//����洢�߽�����ĵ�	
	std::vector<cv::Vec4i> _hierarchy;//����洢��ε�����
	bool _usegaussforegroundfordect;
	std::vector<cv::Rect> _foregroundrects;//

	std::vector<cv::Rect> _pedestrianrects;//��һ��ͼƬ�ϼ�⵽�����˾�������
	std::vector<std::vector<cv::Rect>> _allpedestrianrects;

	std::vector<cv::Rect> _facerects;//��һ��ͼƬ�ϼ�⵽��������������
	std::vector<std::vector<cv::Rect>> _allfacerects;
	std::vector<std::vector<std::vector<cv::Rect>>> _allpedestrianfacerects;

	//std::vector<cv::Mat> _detectface;//��һ��ͼƬ�ϼ�⵽������
	//std::vector<std::vector<cv::Mat>> _alldetectface;

	cv::RotatedRect _trackBox;//����һ����ת�ľ����������CamShift����

	cv::Ptr<cv::FaceRecognizer> _model;//����ģ��
	int _facemodelno;

	cv::VideoCapture _capture;
	cv::VideoWriter _capsave;// ������Ƶ
	bool savevideobool;
};


#endif