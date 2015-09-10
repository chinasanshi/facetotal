
#ifndef CAPTUREFACE_H_
#define CAPTUREFACE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>

using namespace cv;
using namespace std;
class captureface
{
public:
	cv::Mat toGrayscale(cv::Mat src);
	cv::Mat facedect(cv::Mat image, char* cascade_name = "haarcascade_frontalface_alt.xml");
	void takphoto();
	void help();
private:
	cv::VideoCapture _capture;
	cv::Mat _frame;

};


#endif