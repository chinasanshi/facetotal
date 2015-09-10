
//#include <windows.h>
#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//遍历文件图片的时候需要的头文件
#include <map>
#include <fstream>
#include "faceclass.h"

//extern void runvedio(string filename, int& func);
//extern void train_new_model(string fileextension = "*.jpg");
//extern void smartdect(int& func, bool dect_face = false, bool dect_pedestrian = true, bool save_videobool = false, bool showforeground = false, bool use_camshift = true);//参数为人脸模型及该模型的序号
//extern void userdect(int& func, bool dect_face = true, bool dect_pedestrian = true, bool save_videobool = false, bool use_camshift = true);

using namespace cv;
using namespace std;

faceclass::faceclass() {
	_vedio_open = false;
	_facemodelno = 3;
	_func = 1;
}
faceclass::~faceclass() {}

//faceclass.h头文件中有cascade_name的默认值
bool faceclass::addcascade(char* cascade_name) {
	_cascade_name = cascade_name;

	if (!_face_cascade.load(_cascade_name)) {//判断Haar特征加载是否成功

		std::cout << "无法加载级联分类器文件！" << std::endl;
		return false;
	}
	else {
		std::cout << "成功加载级联分类器" << cascade_name << std::endl;
		return true;
	}
}

void faceclass::guassforeground(cv::Mat& image, double learningspeed, bool showforeground) {
	//第二个参数代表背景的更新速率，第三个参数可以设置是否显示背景和前景

	// 运动前景检测，并更新背景;_mog是定义在faceclass中的一个提取背景类BackgroundSubtractorMOG2的对象
	_mog(image, _foreground, learningspeed);//learningspeed为更新速率，默认为0.05，可自己调整
	//去除噪声
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);//膨胀
	erode(_foreground, _foreground, Mat(), Point(-1, -1), 2);//腐蚀
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);

	//_mog.getBackgroundImage(_background);   // 返回当前背景图像

	_showforeground = showforeground;
}

int faceclass::getforegroundrect(bool showforegroundrect) { //参数可以设置是否显示前景运动区域矩形框，只有在前景显示的同时有效
	//每次重新检测都要清空上一次的前景框
	_foregroundrects.clear();
	cv::Mat fgdrect;
	_foreground.copyTo(fgdrect);//复制前景，用于检测轮廓
	fgdrect = fgdrect > 50;//所有像素大于50的像素会设为255，其它的设为0
	//检测前景的轮廓
	findContours(fgdrect, _contours, _hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//检测所有轮廓并重构层次

	//_foregroundrects存储符合条件的外接矩形，由boundingRect函数返回
	int fore_rect_num = 0;
	int idx = 0;//轮廓个数循环
	if (_contours.size()) {//必须加上此判断，否则当视频中只有背景时会出错
		for (; idx >= 0; idx = _hierarchy[idx][0]) {//找到面积最大的轮廓（hierarchy[idx][0]会指向下一个轮廓，若没有下一个轮廓则hierarchy[idx][0]为负数）
			fore_rect_num++;
			if (fabs(contourArea(Mat(_contours[idx]))) > 5000) { //如果当前轮廓的面积大于从前遍历的最大值，则保存当前值	
				
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

				_foregroundrects.push_back(fore_rect);//压栈保存符合条件的外接矩形
			}
			if (fore_rect_num > 5){
				//break;
			}
		}
	}

	if (_showforeground && showforegroundrect) {//如果显示前景，可以设置是否显示前景运动的矩形框
	
		for (vector<Rect>::iterator it = _foregroundrects.begin(); it != _foregroundrects.end(); it++) {//遍历所有符合条件的外接矩形
			rectangle(_foreground, *it, Scalar(255, 255, 255), 3, 8, 0);//在前景画出符合条件的外接矩形框
		}
	}

	if (_showforeground && (!_foreground.empty())) {
		cv::imshow("foreground", _foreground);//显示前景
		//imshow("background", _background);//显示背景
	}

	return _foregroundrects.size();
}

int faceclass::pedestriandect(cv::Mat& image, bool usegaussforegroundfordect) {
	_allpedestrianrects.clear();//重新检测需要清空
	HOGDescriptor hog;//定义HOG对象，采用默认参数
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//采用默认的行人检测分类器
	vector<Rect> pedestrianrects_filtered;//存储检测行人的边界
	size_t pedestrian_no = 0;
	if (usegaussforegroundfordect) {
		for (std::vector<cv::Rect>::iterator itforerects = _foregroundrects.begin(); itforerects != _foregroundrects.end(); ++itforerects) {
			double t1 = (double)getTickCount();//获取系统的时间
			hog.detectMultiScale(Mat(image, *itforerects), _pedestrianrects, 0, cv::Size(8, 8), cv::Size(0, 0), 1.05, 1);//检测行人，采用默认参数好像不行
			size_t i, j;
			for (i = 0; i < _pedestrianrects.size(); ++i) { //遍历regions寻找有没有被嵌套的长方形
				Rect r = _pedestrianrects[i];
				for (j = 0; j < _pedestrianrects.size(); ++j)
					if (j != i && (r & _pedestrianrects[j]) == r)//如果时嵌套的就退出循环
						break;
				if (j == _pedestrianrects.size())
					pedestrianrects_filtered.push_back(r);
			}
			for (j = 0; j < pedestrianrects_filtered.size(); ++j) {
				Rect r = pedestrianrects_filtered[j];
				//HOG检测器返回的矩形框会比真实目标大一些，所以需要将矩形框缩小一些以得到更好的结果
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
			_allpedestrianrects.push_back(pedestrianrects_filtered);//压栈保存此次检测到的行人
			t1 = (double)getTickCount() - t1;//求得检测人脸所需的时间并输出
			pedestrian_no += pedestrianrects_filtered.size();//累加得到所有的行人数目
			std::printf("检测行人的时间为 = %gms/n", t1 * 1000. / ((double)getTickFrequency()));
			std::cout << std::endl;

			//for (vector<Rect>::const_iterator itreg = pedestrianrects_filtered.begin(); itreg != pedestrianrects_filtered.end(); itreg++)//循环画出此次检测到的行人的矩形框
			//{
			//	Rect r = *itreg;
			//	//HOG检测器返回的矩形框会比真实目标大一些，所以需要将矩形框缩小一些以得到更好的结果
			//	r.x += round(r.width*0.1);
			//	r.width = round(r.width*0.8);
			//	r.y += round(r.height*0.07);
			//	r.height = round(r.height*0.8);
			//	rectangle(Mat(image, *itforerects), r, Scalar(0, 0, 255), 3, 8, 0);
			//	//在图片上画出行人区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
			//}

			pedestrianrects_filtered.clear();
			_pedestrianrects.clear();
		}
		return pedestrian_no;
	}
	else {
		double t1 = (double)getTickCount();//获取系统的时间
		hog.detectMultiScale(image, _pedestrianrects, 0, cv::Size(8, 8), cv::Size(0, 0), 1.05, 1);//检测行人，采用默认参数好像不行
		size_t i, j;
		for (i = 0; i < _pedestrianrects.size(); ++i) { //遍历regions寻找有没有被嵌套的长方形
			Rect r = _pedestrianrects[i];
			for (j = 0; j < _pedestrianrects.size(); ++j)
				if (j != i && (r & _pedestrianrects[j]) == r)//如果时嵌套的就退出循环
					break;
			if (j == _pedestrianrects.size())
				pedestrianrects_filtered.push_back(r);
		}
		for (j = 0; j < pedestrianrects_filtered.size(); ++j) {
			Rect r = pedestrianrects_filtered[j];
			//HOG检测器返回的矩形框会比真实目标大一些，所以需要将矩形框缩小一些以得到更好的结果
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
		_allpedestrianrects.push_back(pedestrianrects_filtered);//压栈保存此次检测到的行人
		t1 = (double)getTickCount() - t1;//求得检测人脸所需的时间并输出
		std::printf("检测行人的时间为 = %gms/n", t1 * 1000. / ((double)getTickFrequency()));
		std::cout << std::endl;

		pedestrian_no = pedestrianrects_filtered.size();//检测到的行人个数
		pedestrianrects_filtered.clear();
		_pedestrianrects.clear();

		return pedestrian_no;
	}

	if (_allpedestrianrects.empty()) {
		std::cout << "未检测到行人，从而无法进行人脸检测！" << std::endl;
		return 0;
	}
}

int faceclass::facedect(cv::Mat& image, bool usegaussforegroundfordect) {    //检测一张图片里的人脸
//第一个参数为待检测的图片，第二个参数为是否在前景运动框内检测人脸；返回值代表检测的人脸个数(但并不代表具体个数)
//检测完成后，检测到的人脸矩形框存储在_allfacerects中

	_allfacerects.clear();//重新检测时需要清除原来存储的人脸
	_facerects.clear();
	_allface_id.clear();
	_usegaussforegroundfordect = usegaussforegroundfordect;//将此布尔值传递给类的成员变量
	//如果不使用前景信息
	if (!_usegaussforegroundfordect) {
		_face_cascade.detectMultiScale(image, _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		if (_facerects.empty()) {
			std::cout << "没有检测到人脸！" << std::endl;
			return 0;
		}
		else {
			for (int fNo = 0; fNo < int(_facerects.size()); fNo++) {   //循环画出人脸的矩形框
				cv::rectangle(image, _facerects[fNo], Scalar(255, 0, 255), 3, 8, 0);
				//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
			}
			std::cout << "检测到" << _facerects.size() << "张人脸！" << std::endl;
			_allfacerects.push_back(_facerects);
			return _facerects.size();
		}
	}
	else { //如果使用前景信息

		size_t facenum = 0;//记录总共检测到的人脸个数

		for (int ROINo = 0; ROINo < _foregroundrects.size(); ROINo++) {
			double t = (double)cvGetTickCount();//获取系统的时间
			//_frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			_face_cascade.detectMultiScale(cv::Mat(image, _foregroundrects[ROINo]), _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			facenum += _facerects.size();
			//如果检测到人脸就保存
			if (_foregroundrects.size()) {
				_allfacerects.push_back(_facerects);//压栈保存此次检测到的人脸
			}
			t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
			printf("检测人脸的时间为 = %gms/n", t / ((double)cvGetTickFrequency() * 1000.));
			cout << endl;

			for (int fNo = 0; fNo < int(_facerects.size()); fNo++) { //循环画出人脸的矩形框
				cv::rectangle(image, Rect{ _foregroundrects[ROINo].x + _facerects[fNo].x, _foregroundrects[ROINo].y + _facerects[fNo].y, _facerects[fNo].width, _facerects[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
				//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
			}
		}

		if (_allfacerects.size() == 0) {
			std::cout << "没有检测到人脸！" << std::endl;
			return 0;
		}
		else {
			std::cout << "检测到的人脸区域保存到了成员变量'_allfacerects'里面！\n共检测到" << facenum << "张人脸" << std::endl;
			return facenum;
		}
	}
}


//函数功能：遍历该路径下的所有符合命名规则的以参数为扩展名的图片，默认为".jpg"
bool faceclass::traversal(string fileextension)//filename允许有通配符，'？'代表一个字符，'*'代表0到任意字符
{
	int jpgNum = 0;//记录总共找到的图片的个数
	char labelchar[10] = "\0";//保存从图片名字里面提取到的标号
	int label = 0;
	_finddata_t fileInfo;//定义结构体
	long handle = _findfirst(fileextension.c_str(), &fileInfo);//寻找第一个图片文件

	if (handle == -1L)//如果没有找到文件则句柄返回-1L
	{
		cerr << "未找到图片文件！" << std::endl;
		return false;
	}

	do {
		if (fileInfo.name[0] != 'f')//如果图片名称不是以f开头则不符合条件，进入下一次查找
		{
			continue;//不可用break，否则会直接退出do-while循环
		}
		for (unsigned int i = 0; i < strlen(fileInfo.name); i++) {
			if (fileInfo.name[i + 1] == '_')//遇到下划线则退出for循环
			{
				labelchar[i] = '\0';
				break;
			}
			labelchar[i] = fileInfo.name[i + 1];//图片名字的第一个字母为f，不用保留
		}
		label = atoi(labelchar);//将char型标号转换成int型
		_labels.push_back(label);//将标号压栈传入标号容器
		cv::Mat face_picture = imread(fileInfo.name, 0);
		resize(face_picture, face_picture, cv::Size{ 200, 200 });
		_faces.push_back(imread(fileInfo.name, 0));//将找到的图片以灰度形式压栈出入faces，否则不会有结果
		jpgNum++;//图片数加1
		std::cout << fileInfo.name << std::endl;//输出找到的图片的名字
	} while (_findnext(handle, &fileInfo) == 0);//寻找下一张图片
	std::cout << " 有效 .jpg 图片文件的个数为:  " << jpgNum << std::endl;//输出找到的图片的个数

	return true;
}

bool faceclass::setmodelno(int modelno)
//默认创建LBP人脸识别模型，可以通过参数修改
{
	_facemodelno = modelno;

	switch (_facemodelno)
	{
	case 1:
		_model = createEigenFaceRecognizer(10);//设置为PCA模型
		std::cout << "设置为PCA人脸识别模型！" << std::endl;
		return true;
		break;
	case 2:
		_model = createFisherFaceRecognizer();//设置为FisherFace模型
		std::cout << "设置为FisherFace人脸识别模型！" << std::endl;
		return true;
		break;
	case 3:
		_model = createLBPHFaceRecognizer(1,8,8,8,123.0);//设置为LBP模型，此模型检测人脸效果最好
		//_model->set("threshold", 80);
		std::cout << "设置为LBP人脸识别模型！" << std::endl;
		return true;
		break;
	default:
		std::cout << "输入的参数不符合要求，设置人脸识别模型失败！" << std::endl;
		return false;
		break;
	}
}

void faceclass::trainsavefacemodel()
{
	_model->train(_faces, _labels);//训练模型
	switch (_facemodelno) {
	case 1:
		_model->save("PCAFace.xml");//将训练好的人脸PCA模型保存成XML文件
		std::cout << "保存PCA人脸识别模型！" << std::endl;
		break;
	case 2:
		_model->save("FisherFace.xml");//将训练好的人脸FisherFace模型保存成XML文件
		std::cout << "保存FisherFace人脸识别模型！" << std::endl;
		break;
	case 3:
		_model->save("LBPHface.xml");//将训练好的人脸LBP模型保存成XML文件
		std::cout << "保存LBP人脸识别模型！" << std::endl;
		break;
	default:
		std::cout << "输入的参数不符合要求，保存人脸识别模型失败！" << std::endl;
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
		_model->load("PCAFace.xml");//将训练好的人脸PCA模型的XML文件加载到模型
		std::cout << "加载PCA人脸识别模型！" << std::endl;
		break;
	case 2:
		_model->load("FisherFace.xml");//将训练好的人脸FisherFace模型的XML文件加载到模型
		std::cout << "加载FisherFace人脸识别模型！" << std::endl;
		break;
	case 3:
		_model->load("LBPHface.xml");//将训练好的人脸LBP模型的XML文件加载到模型
		std::cout << "加载LBP人脸识别模型！" << std::endl;
		break;
	default:
		std::cout << "输入的参数不符合要求，加载人脸识别模型失败！" << std::endl;
		break;
	}
}

void faceclass::facecamshift(cv::Mat& image, int face_id) {
	cv::Rect trackWindow;//定义跟踪的矩形
	//RotatedRect trackBox;//定义一个旋转的矩阵类对象，由CamShift返回
	int hsize = 16;//每一维直方图的大小
	float hranges[] = { 0, 180 };//hranges在后面的计算直方图函数中要用到
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//
	cv::Mat hsv, hue, mask, hist, backproj;
	int facenumforid = 0;
	//camshift跟踪人脸
	for (std::vector<std::vector<cv::Rect>>::iterator itallf = _allfacerects.begin(); itallf != _allfacerects.end(); ++itallf) {
		for (std::vector<cv::Rect>::iterator itf = itallf->begin(); itf != itallf->end(); ++itf) {
			cv::cvtColor(image, hsv, CV_BGR2HSV);//将rgb摄像头帧转化成hsv空间的
			//inRange函数的功能是检查输入数组每个元素大小是否在2个给定数值之间，可以有多通道,mask保存0通道的最小值，也就是h分量
			//这里利用了hsv的3个通道，比较h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)。如果3个通道都在对应的范围内，则
			//mask对应的那个点的值全为1(0xff)，否则为0(0x00).
			cv::inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
			int ch[] = { 0, 0 };
			hue.create(hsv.size(), hsv.depth());//hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的，红绿蓝之间相差120度，反色相差180度
			cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);//将hsv第一个通道(也就是色调)的数复制到hue中，0索引数组

			//此处的构造函数roi用的是Mat hue的矩阵头，且roi的数据指针指向hue，即共用相同的数据，select为其感兴趣的区域
			trackWindow = *itf;
			cv::Mat roi(hue, trackWindow), maskroi(mask, trackWindow);//mask保存的hsv的最小值

			//calcHist()函数第一个参数为输入矩阵序列，第2个参数表示输入的矩阵数目，第3个参数表示将被计算直方图维数通道的列表，第4个参数表示可选的掩码函数
			//第5个参数表示输出直方图，第6个参数表示直方图的维数，第7个参数为每一维直方图数组的大小，第8个参数为每一维直方图bin的边界
			cv::calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);//将roi的0通道计算直方图并通过mask放入hist中，hsize为每一维直方图的大小
			cv::normalize(hist, hist, 0, 255, CV_MINMAX);//将hist矩阵进行数组范围归一化，都归一化到0~255

			cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);//计算直方图的反向投影，计算hue图像0通道直方图hist的反向投影，并让入backproj中
			backproj &= mask;

			//opencv2.0以后的版本函数命名前没有cv两字了，并且如果函数名是由2个意思的单词片段组成的话，且前面那个片段不够成单词，则第一个字母要
			//大写，比如Camshift，如果第一个字母是个单词，则小写，比如meanShift，但是第二个字母一定要大写
			_trackBox = cv::CamShift(backproj, trackWindow,               //trackWindow为鼠标选择的区域，TermCriteria为确定迭代终止的准则
			                         cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));//CV_TERMCRIT_EPS是通过forest_accuracy,CV_TERMCRIT_ITER

			if (trackWindow.area() <= 1) {                                                 //是通过max_num_of_trees_in_the_forest
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
				                   trackWindow.x + r, trackWindow.y + r) &
				              Rect(0, 0, cols, rows);//Rect函数为矩阵的偏移和大小，即第一二个参数为矩阵的左上角点坐标，第三四个参数为矩阵的宽和高
			}

			//cv::ellipse(image, _trackBox, Scalar(255, 255, 0), 3, CV_AA);//跟踪的时候以椭圆为代表目标
			Rect facerecttodraw;
			facerecttodraw.x = _trackBox.center.x - _trackBox.size.width / 2;
			facerecttodraw.y = _trackBox.center.y - _trackBox.size.height / 2;
			facerecttodraw.width = _trackBox.size.width;
			facerecttodraw.height = _trackBox.size.height;
			putText(image, _id_dict[_allface_id[facenumforid]], cv::Point(facerecttodraw.x, facerecttodraw.y), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));//在图片中输出字符
			cv::rectangle(image, facerecttodraw, Scalar(255, 255, 0), 3, 0);

			facenumforid++;

		}
	}
}

//默认打开摄像头，若要打开文件需要将第一个参数设为-2，并将文件名传递给第二个参数
bool faceclass::opencamera(string filename, int cameranum) {
	//if (cameranum == -2 && filename != "nothing"){
	if (filename != "nothing") {
		_capture.open(filename);
	}
	else if (cameranum != -2) {
		_capture.open(cameranum);
	}

	if (!_capture.isOpened()) {
		std::cout << "打开视频文件失败！" << std::endl;
		return false;
	}
	else {
		std::cout << "视频文件已打开！" << std::endl;
		return true;
	}
}

void faceclass::runvedio(string filename, int& func) {
	//if (opencamera(filename)){
	//	_vedio_open = true;

	//while (_capture.read(_frame)){
	//	char c = waitKey(33);//控制帧率
	//	cv::imshow("vedio", _frame);//显示视频
	//}
	//}

	if (func == 1) {
		while (_capture.read(_frame) && func == 1) {
			char c = waitKey(33);//控制帧率

			if (c == 'q') {
				break;
			}

			cv::imshow("vedio", _frame);//显示视频

			//if (_func != 1){
			//	break;
			//}
		}
	}
}

//存储标签号与名字的对应关系
void faceclass::insertdict(int lablenum, string name) {
	_id_dict.insert(pair<int, string>(lablenum, name));
}

//按键‘p’则预测人脸，按键‘q’退出人脸预测
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

	//if (opencamera()){//如果成功打开摄像头
	//if (!_frame.empty()){//如果成功打开摄像头
	if (func == 2) {

		//if (savevideobool){
		//	savevideoinit();
		//}

		size_t frame_no = 0;
		cv::Mat face;//保存人脸
		fstream outfile;
		char timetmp[20];
		bool ispre = false;
		string record = "record.txt";
		int predictedLabel = 0;
		int facenum = 0;
		while (_capture.read(_frame) && func == 2) {
			char c = waitKey(33);//控制帧率

			if (c == 'q') {
				break;
			}

			if (frame_no % 30 == 0 ) {
				guassforeground(_frame, 0.05, showforeground);
				int forenum = getforegroundrect();
				if (forenum) {
					if (dect_pedestrian) { //如果检测行人
						pedestriandect(_frame, true);//在当前帧中检测行人，此函数内永远使用前景信息（第二个参数为true）
					}
					if (dect_face) {
						int facenum = facedect(_frame, true);//在当前帧中检测人脸，此函数内永远使用前景信息（第二个参数为true）
						for (std::vector<std::vector<cv::Rect>>::iterator itallrects = _allfacerects.begin(); itallrects != _allfacerects.end(); ++itallrects) {
							for (std::vector<cv::Rect>::iterator itrects = itallrects->begin(); itrects != itallrects->end(); ++itrects) {
								face = cv::Mat(_frame, *itrects);
								resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像，确保与模型训练时采取的图片大小相同
								cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
								cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);
								predictedLabel = _model->predict(face);//利用模型识别人脸ID
								_id = predictedLabel;
								_allface_id.push_back(_id);
								//cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//将检测到的人脸显示出来
								std::cout << "检测到是" << _id_dict[predictedLabel] << std::endl;
								string face_id = _id_dict[predictedLabel];
								putText(_frame, face_id, cv::Point(itrects->x, itrects->y), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));//在图片中输出字符

								time_t qiandaotime = time(0);//获取系统当前的时间
								strftime(timetmp, 20, "%Y/%m/%d %H:%M:%S", localtime(&qiandaotime));
								//将记录保存到record.txt文件中
								outfile.open(record, ios::app);//以追加的方式写文件
								outfile << "北京时间 " << timetmp << " " << _id_dict[predictedLabel] << "出现在监控区域" << endl;//将签到的人标号和时间记录下来
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
				_capsave << _frame;//保存视频帧
			}

			cv::imshow("predect", _frame);//显示视频

			frame_no++;

			//if (_func != 2){
			//	break;
			//}
		}
	}
}

//如果不使用背景检测方法，每10帧视频检测一次
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

	//if (opencamera()){//如果成功打开摄像头
	//if (!_frame.empty()){//如果成功打开摄像头
	if (func == 3) {

		//if (savevideobool){
		//	savevideoinit();
		//}

		size_t frame_no = 0;
		cv::Mat face;//保存人脸
		fstream outfile;
		char timetmp[20];
		bool ispre = false;
		string record = "record.txt";
		int predictedLabel = 0;
		int facenum = 0;
		while (_capture.read(_frame) && func == 3) {
			char c = waitKey(33);//控制帧率

			if (c == 'q') {
				break;
			}

			if (dect_pedestrian && frame_no % 5 == 0) { //如果检测行人
				pedestriandect(_frame, false);//在当前帧中检测行人，此函数内永远不使用前景信息（第二个参数为false）
			}
			if (dect_face && frame_no % 30 == 0) {
				int facenum = facedect(_frame, false);//在当前帧中检测人脸，此函数内永远不使用前景信息（第二个参数为false）
				for (std::vector<std::vector<cv::Rect>>::iterator itallrects = _allfacerects.begin(); itallrects != _allfacerects.end(); ++itallrects) {
					for (std::vector<cv::Rect>::iterator itrects = itallrects->begin(); itrects != itallrects->end(); ++itrects) {
						face = cv::Mat(_frame, *itrects);
						resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像，确保与模型训练时采取的图片大小相同
						cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
						cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);
						predictedLabel = _model->predict(face);
						_id = predictedLabel;
						_allface_id.push_back(_id);

						//cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//将检测到的人脸显示出来
						std::cout << "检测到是" << _id_dict[predictedLabel] << std::endl;
						string face_id = _id_dict[predictedLabel];
						putText(_frame, face_id, cv::Point(itrects->x, itrects->y), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));//在图片中输出字符
						time_t qiandaotime = time(0);//获取系统当前的时间
						strftime(timetmp, 20, "%Y/%m/%d %H:%M:%S", localtime(&qiandaotime));
						//将记录保存到record.txt文件中
						outfile.open(record, ios::app);//以追加的方式写文件
						outfile << "北京时间： " << timetmp << " " << _id_dict[predictedLabel] << "出现在监控区域" << endl;//将检测到的到的人标号和时间记录下来
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
				_capsave << _frame;//保存视频帧
			}

			cv::imshow("predect", _frame);//显示视频

			frame_no++;

			//if (_func != 3){
			//	break;
			//}
		}
	}
}


bool faceclass::savevideoinit() { //初始化保存视频功能
	_capture >> _frame;//读取一帧的视频文件
	_capsave.open("监控.avi", CV_FOURCC('M', 'J', 'P', 'G'), 33, _frame.size(), 1);
	_capsave << _frame;//保存一帧的视频文件
	if (!_capsave.isOpened()) { //判断保存视频是否正确初始化
		cout << "保存视频失败!" << endl;
		return false;
	}
	else {
		std::cout << "保存视频函数初始化成功！" << std::endl;
		return true;
	}
}


