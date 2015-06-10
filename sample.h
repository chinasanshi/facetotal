
#ifndef SAMPLE_H_
#define SAMLLE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>
#include <string>
#include <map>

using namespace cv;
using namespace std;
class sample
{
public:
	sample();
	~sample();
	bool addcascade(char* cascade_name ="haarcascade_frontalface_alt.xml");

	cv::Mat facedect(cv::Mat image);
	bool opencamera(string filename = "nothing", int cameranum = 0);
	void runvedio(string filename);//播放视频
	void takephoto(int& func, string labelcin, string sample_name, string sample_no);
	void help();

	int _func;
private:
	char* _cascade_name;
	cv::CascadeClassifier _frontalface_cascade;
	cv::Rect _face_rect;
	cv::VideoCapture _capture;
	cv::Mat _frame;
	int _sample_label;//样本标签号
	string _sample_name;//样本名称
	int _sample_no;//样本序号
	map<int, string> _id_dict;

};


#endif