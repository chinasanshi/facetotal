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
#include <map>
#include <io.h>//遍历文件图片的时候需要的头文件
//#include<conio.h>

using namespace cv;
using namespace std;

class faceclass
{
public:

	explicit faceclass();
	~faceclass();
	bool addcascade(char* cascade_name = "haarcascade_frontalface_alt.xml");//加载Haar特征级联分类器
	void guassforeground(cv::Mat& image, double learningspeed = 0.05, bool showforeground = false);
	int getforegroundrect(bool showforegroudrect = true);

	int pedestriandect(cv::Mat& image, bool usegaussforegroundfordect = true);//检测行人

	int facedect(cv::Mat& image, bool usegaussforegroundfordect = true);//检测一张图片里的人脸


	bool cheakinputisnum(string inputlable);//检测输入的标号是否是有效地数字
	string imagenamegen(int facelable, int facenum);
	void takphoto();//采集人脸

	bool traversal(string fileextension = "*.jpg");//遍历该路径下的所有符合命名规则的以参数为扩展名的图片
	void trainsavefacemodel();
	void train_new_model(string fileextension = "*.jpg");
	void loadfacemodel();

	void facecamshift(cv::Mat& image, int face_id);

	void insertdict(int lablenum, string name);//添加标签号与姓名的对应关系

	bool setmodelno(int modelno = 3);
	bool opencamera(string filename = "nothing", int cameranum = 0);
	void runvedio(string filename, int& func);//播放视频
	void smartdect(int& func, bool dect_face = false, bool dect_pedestrian = true, bool save_videobool = false, bool showforeground = false, bool use_camshift = true);//参数为人脸模型及该模型的序号
	void userdect(int& func, bool dect_face = true, bool dect_pedestrian = true, bool save_videobool = false, bool use_camshift = true);
	bool savevideoinit();


	int _func;

private:

	bool _vedio_open;
	std::vector<cv::Mat> _faces;// 保存图片的容器和相应标号的容器
	std::vector<int> _labels;

	cv::CascadeClassifier _face_cascade;
	char* _cascade_name;
	
	cv::Mat _frame;
	cv::Mat _image;

	cv::Mat _foreground;//保存前景帧数据
	cv::Mat _background;//保存背景帧数据
	cv::BackgroundSubtractorMOG2 _mog;
	bool _showforeground;
	std::vector<std::vector<cv::Point> > _contours;//定义存储边界所需的点	
	std::vector<cv::Vec4i> _hierarchy;//定义存储层次的向量
	bool _usegaussforegroundfordect;
	std::vector<cv::Rect> _foregroundrects;//

	std::vector<cv::Rect> _pedestrianrects;//从一张图片上检测到的行人矩形区域
	std::vector<std::vector<cv::Rect>> _allpedestrianrects;

	std::vector<cv::Rect> _facerects;//从一张图片上检测到的人脸矩形区域
	std::vector<std::vector<cv::Rect>> _allfacerects;
	std::vector<std::vector<std::vector<cv::Rect>>> _allpedestrianfacerects;

	//std::vector<cv::Mat> _detectface;//从一张图片上检测到的人脸
	//std::vector<std::vector<cv::Mat>> _alldetectface;

	cv::RotatedRect _trackBox;//定义一个旋转的矩阵类对象，由CamShift返回

	cv::Ptr<cv::FaceRecognizer> _model;//人脸模型
	int _facemodelno;

	map<int, string> _id_dict;
	int _id;
	vector<int> _allface_id;

	cv::VideoCapture _capture;
	cv::VideoWriter _capsave;// 保存视频
	bool savevideobool;
};


#endif