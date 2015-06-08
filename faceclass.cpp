//
//
#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

//#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//遍历文件图片的时候需要的头文件

#include "faceclass.h"

using namespace cv;
using namespace std;

faceclass::faceclass(){}
faceclass::~faceclass(){}


bool faceclass::addcascade(char* cascade_name)
{
	_cascade_name = cascade_name;

	if (!_face_cascade.load(_cascade_name))//判断Haar特征加载是否成功
	{
		std::cout << "无法加载级联分类器文件！" << std::endl;
		return false;
	}
	else
	{
		std::cout << "加载级联分类器成功！" << std::endl;
		return true;
	}
}

void faceclass::guassforeground(cv::Mat& image, double learningspeed, bool showforeground)
//第一个参数代表背景的更新速率，第二个参数可以设置是否显示背景和前景
{
	// 运动前景检测，并更新背景
	_mog(image, _foreground, learningspeed);//learningspeed为更新速率，默认为0.05，可自己调整
	//去除噪声
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);//膨胀
	erode(_foreground, _foreground, Mat(), Point(-1, -1), 2);//腐蚀
	dilate(_foreground, _foreground, Mat(), Point(-1, -1), 1);

	//_mog.getBackgroundImage(_background);   // 返回当前背景图像

	_showforeground = showforeground;
	if (_showforeground && (!_foreground.empty()))
	{
		imshow("foreground", _foreground);//显示前景
		//imshow("background", _background);//显示背景
	}
}

void faceclass::getforegroundrect(bool showforegroundrect)//参数可以设置是否显示前景运动区域矩形框
{
	cv::Mat fgdrect;
	_foreground.copyTo(fgdrect);//复制前景，用于检测轮廓
	fgdrect = fgdrect > 50;//所有像素大于50的像素会设为255，其它的设为0
	//检测前景的轮廓
	//findContours(fgdrect, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//只检测最外面的轮廓
	//findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//检测所有轮廓并分为两层
	findContours(fgdrect, _contours, _hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//检测所有轮廓并重构层次

	//_foregroundrects存储符合条件的外接矩形，由boundingRect函数返回
	int idx = 0;//轮廓个数循环
	if (_contours.size())//必须加上此判断，否则当视频中只有背景时会出错
	{
		for (; idx >= 0; idx = _hierarchy[idx][0])//找到面积最大的轮廓（hierarchy[idx][0]会指向下一个轮廓，若没有下一个轮廓则hierarchy[idx][0]为负数）
		{
			if (fabs(contourArea(Mat(_contours[idx]))) > 5000)//如果当前轮廓的面积大于从前遍历的最大值，则保存当前值
			{
				_foregroundrects.push_back(boundingRect(_contours[idx]));//压栈保存符合条件的外接矩形
			}
		}
	}

	if (_showforeground && showforegroundrect)//如果显示前景，可以设置是否显示前景运动的矩形框
	{
		for (vector<Rect>::iterator it = _foregroundrects.begin(); it != _foregroundrects.end(); it++)//遍历所有符合条件的外接矩形
		{
			rectangle(_foreground, *it, Scalar(255, 0, 255), 3, 8, 0);//在前景画出符合条件的外接矩形框
		}
	}
}

void faceclass::pedestriandect(cv::Mat& image, bool showpedestrianrect)
{
	_allpedestrianrects.clear();//重新检测需要清空
	HOGDescriptor hog;//定义HOG对象，采用默认参数
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//采用默认的行人检测分类器
	vector<Rect> pedestrianrects_filtered;//存储检测行人的边界
	for (std::vector<cv::Rect>::iterator itforerects = _foregroundrects.begin(); itforerects != _foregroundrects.end(); ++itforerects)
	{
		double t1 = (double)getTickCount();//获取系统的时间
		hog.detectMultiScale(Mat(image, *itforerects), _pedestrianrects, 0, cv::Size(8, 8), cv::Size(0, 0), 1.05, 1);//检测行人，采用默认参数好像不行
		size_t i, j;
		for (i = 0; i < _pedestrianrects.size(); i++)//遍历regions寻找有没有被嵌套的长方形
		{
			Rect r = _pedestrianrects[i];
			for (j = 0; j < _pedestrianrects.size(); j++)
				if (j != i && (r & _pedestrianrects[j]) == r)//如果时嵌套的就退出循环
					break;
			if (j == _pedestrianrects.size())
				pedestrianrects_filtered.push_back(r);
		}
		for (i = 0; i < pedestrianrects_filtered.size(); i++)
		{
			Rect r = pedestrianrects_filtered[i];
			//HOG检测器返回的矩形框会比真实目标大一些，所以需要将矩形框缩小一些以得到更好的结果
			r.x += round(r.width*0.1);
			r.width = round(r.width*0.8);
			r.y += round(r.height*0.07);
			r.height = round(r.height*0.8);
			rectangle(image, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}
		_allpedestrianrects.push_back(pedestrianrects_filtered);//压栈保存此次检测到的行人
		t1 = (double)getTickCount() - t1;//求得检测人脸所需的时间并输出
		printf("检测行人的时间为 = %gms/n", t1*1000. / ((double)getTickFrequency()));
		std::cout << std::endl;

		if (showpedestrianrect)
		{
			for (vector<Rect>::const_iterator itreg = pedestrianrects_filtered.begin(); itreg != pedestrianrects_filtered.end(); itreg++)//循环画出此次检测到的行人的矩形框
			{
				rectangle(Mat(image, *itreg), *itreg, Scalar(0, 0, 255), 3, 8, 0);
				//在图片上画出行人区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
			}
		}
		pedestrianrects_filtered.clear();
		_pedestrianrects.clear();
	}
	if (_allpedestrianrects.empty())
	{
		std::cout << "未检测到行人，从而无法进行人脸检测！" << std::endl;
	}
}

int faceclass::facedect(cv::Mat& image, bool usegaussforegroundfordect)//检测一张图片里的人脸
//第一个参数为待检测的图片，第二个参数为是否在前景运动框内检测人脸；返回值代表检测的人脸个数(但并不代表具体个数)
//检测完成后，检测到的人脸矩形框存储在_allfacerects中
{
	_allfacerects.clear();//从新检测时需要清除原来存储的人脸
	_facerects.clear();

	_usegaussforegroundfordect = usegaussforegroundfordect;//将此布尔值传递给类的成员变量
	if (!_usegaussforegroundfordect)
	{
		_face_cascade.detectMultiScale(image, _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		if (_facerects.empty())
		{
			std::cout << "没有检测到人脸！" << std::endl;
			return 0;
		}
		else if (_facerects.size() == 1)
		{
			std::cout << "检测一张到人脸！" << std::endl;
			_allfacerects.push_back(_facerects);
			return 1;
		}
		else
		{
			std::cout << "检测到多张人脸！" << std::endl;
			_allfacerects.push_back(_facerects);
			return 2;
		}
	}
	else
	{
		for (int ROINo = 0; ROINo < _foregroundrects.size(); ROINo++)
		{
			double t = (double)cvGetTickCount();//获取系统的时间
			//_frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			_face_cascade.detectMultiScale(cv::Mat(image, _foregroundrects[ROINo]), _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

			_allfacerects.push_back(_facerects);//压栈保存此次检测到的人脸
			t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
			printf("检测人脸的时间为 = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
			cout << endl;

			for (int fNo = 0; fNo < int(_facerects.size()); fNo++)//循环画出人脸的矩形框
			{
				rectangle(image, Rect{ _foregroundrects[ROINo].x + _facerects[fNo].x, _foregroundrects[ROINo].y + _facerects[fNo].y, _facerects[fNo].width, _facerects[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
				//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
				//用椭圆标出人脸
				//Point center(rects[ROINo].x + faces[fNo].x + faces[fNo].width*0.5, rects[ROINo].y + faces[fNo].y + faces[fNo].height*0.5);//椭圆中心点坐标要加上ROI区域的左上角坐标点
				//ellipse(frame, center, Size(faces[fNo].width*0.5, faces[fNo].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 3, 8, 0);
			}
		}

		if (_allfacerects.size() == 0)
		{
			std::cout << "没有检测到人脸！" << std::endl;
			return 0;
		}
		else
		{
			std::cout << "检测到的人脸区域保存到了成员变量“_allfacerects”里面！" << std::endl;
			return 3;
		}
	}
}

void faceclass::facedect(cv::Mat& image, int i)
{
	pedestriandect(image);//检测图片中的行人

	for (std::vector<std::vector<cv::Rect>>::iterator itallpeddestrian = _allpedestrianrects.begin(); itallpeddestrian != _allpedestrianrects.end(); ++itallpeddestrian)
	{
		for (std::vector<cv::Rect>::iterator itpeddestrian = itallpeddestrian->begin(); itpeddestrian != itallpeddestrian->end(); ++itpeddestrian)
		{
			double t = (double)cvGetTickCount();//获取系统的时间
			//_frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
			_face_cascade.detectMultiScale(cv::Mat(image, *itpeddestrian), _facerects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

			_allfacerects.push_back(_facerects);//压栈保存此次检测到的人脸
			t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
			printf("检测人脸的时间为 = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
			cout << endl;

			for (int fNo = 0; fNo < int(_facerects.size()); fNo++)//循环画出人脸的矩形框
			{
				rectangle(image, Rect{ (*itpeddestrian).x + _facerects[fNo].x, (*itpeddestrian).y + _facerects[fNo].y, _facerects[fNo].width, _facerects[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
				//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
				//用椭圆标出人脸
				//Point center(rects[ROINo].x + faces[fNo].x + faces[fNo].width*0.5, rects[ROINo].y + faces[fNo].y + faces[fNo].height*0.5);//椭圆中心点坐标要加上ROI区域的左上角坐标点
				//ellipse(frame, center, Size(faces[fNo].width*0.5, faces[fNo].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 3, 8, 0);
			}
		}

		_allpedestrianfacerects.push_back(_allfacerects);
		if (_allpedestrianfacerects.size() == 0)
		{
			std::cout << "没有检测到人脸！" << std::endl;
		}
		else
		{
			std::cout << "检测到的人脸区域保存到了成员变量“_allpedestrianfacerects”里面！" << std::endl;
		}
	}
}

//只允许为灰度图，否则报错，然后将灰度图归一化
Mat& faceclass::toGrayscale(Mat& src)
{
	// 只允许为单通道的
	if (src.channels() != 1)
	{
		CV_Error(CV_StsBadArg, "只支持单通道的矩阵！");
	}
	// 创建并返回归一化后的图片
	//Mat dst;
	cv::normalize(src, src, 0, 255, NORM_MINMAX, CV_8UC1);
	return src;
}

/*结构体说明
struct _finddata_t
{
unsigned attrib;     //文件属性
time_t time_create;  //文件创建时间
time_t time_access;  //文件上一次访问时间
time_t time_write;   //文件上一次修改时间
_fsize_t size;  //文件字节数
char name[_MAX_FNAME]; //文件名
};
//按FileName命名规则匹配当前目录第一个文件
_findfirst(_In_ const char * FileName, _Out_ struct _finddata64i32_t * _FindData);
//按FileName命名规则匹配当前目录下一个文件
_findnext(_In_ intptr_t _FindHandle, _Out_ struct _finddata64i32_t * _FindData);
//关闭_findfirst返回的文件句柄
_findclose(_In_ intptr_t _FindHandle);*/

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

	do
	{
		if (fileInfo.name[0] != 'f')//如果图片名称不是以f开头则不符合条件，进入下一次查找
		{
			continue;//不可用break，否则会直接退出do-while循环
		}
		for (unsigned int i = 0; i < strlen(fileInfo.name); i++)
		{
			if (fileInfo.name[i + 1] == '_')//遇到下划线则退出for循环
			{
				labelchar[i] = '\0';
				break;
			}
			labelchar[i] = fileInfo.name[i + 1];//图片名字的第一个字母为f，不用保留
		}
		label = atoi(labelchar);//将char型标号转换成int型
		_labels.push_back(label);//将标号压栈传入标号容器
		_faces.push_back(imread(fileInfo.name, 0));//将找到的图片以灰度形式压栈出入faces，否则不会有结果
		jpgNum++;//图片数加1
		std::cout << fileInfo.name << std::endl;//输出找到的图片的名字
	} while (_findnext(handle, &fileInfo) == 0);//寻找下一张图片
	std::cout << " 有效 .jpg 图片文件的个数为:  " << jpgNum << std::endl;//输出找到的图片的个数

	return true;
}

void faceclass::trainsavefacemodel()
{
	_model->train(_faces, _labels);//训练模型
	switch (_facemodelno)
	{
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

void faceclass::loadfacemodel()
{
	switch (_facemodelno)
	{
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

void faceclass::facecamshift(cv::Mat& image)
{
	cv::Rect trackWindow;//定义跟踪的矩形
	//RotatedRect trackBox;//定义一个旋转的矩阵类对象，由CamShift返回
	int hsize = 16;//每一维直方图的大小
	float hranges[] = { 0, 180 };//hranges在后面的计算直方图函数中要用到
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//
	cv::Mat hsv, hue, mask, hist, backproj;

	//camshift跟踪人脸
	for (std::vector<std::vector<cv::Rect>>::iterator itallf = _allfacerects.begin(); itallf != _allfacerects.end(); ++itallf)
	{
		for (std::vector<cv::Rect>::iterator itf = itallf->begin(); itf != itallf->end(); ++itf)
		{
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

			if (trackWindow.area() <= 1)                                                  //是通过max_num_of_trees_in_the_forest  
			{
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
					trackWindow.x + r, trackWindow.y + r) &
					Rect(0, 0, cols, rows);//Rect函数为矩阵的偏移和大小，即第一二个参数为矩阵的左上角点坐标，第三四个参数为矩阵的宽和高
			}

			cv::ellipse(image, _trackBox, Scalar(255, 255, 0), 3, CV_AA);//跟踪的时候以椭圆为代表目标
		}
	}
}

bool faceclass::setmodelno(int modelno)
//默认创建LBP人脸识别模型，可以通过参数修改
{
	_facemodelno = modelno;

	switch (_facemodelno)
	{
	case 1:
		_model = createEigenFaceRecognizer(10);//创建PCA模型
		std::cout << "创建PCA人脸识别模型！" << std::endl;
		return true;
		break;
	case 2:
		_model = createFisherFaceRecognizer();//创建FisherFace模型
		std::cout << "创建FisherFace人脸识别模型！" << std::endl;
		return true;
		break;
	case 3:
		_model = createLBPHFaceRecognizer();//创建LBP模型，此模型检测人脸效果最好
		std::cout << "创建LBP人脸识别模型！" << std::endl;
		return true;
		break;
	default:
		std::cout << "输入的参数不符合要求，创建人脸识别模型失败！" << std::endl;
		return false;
		break;
	}
}

bool faceclass::opencamera()
{
	_capture.open(0);
	if (!_capture.isOpened())
	{
		std::cout << "打开摄像头失败！" << std::endl;
		return false;
	}
	else
	{
		std::cout << "摄像头已打开！" << std::endl;
		return true;
	}
}

//按键‘p’则预测人脸，按键‘q’退出人脸预测
void faceclass::predect(bool usepedestrianrects, bool savevideobool)
{
	if (opencamera())//如果成功打开摄像头
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

		cv::Mat temp;//保存视频帧
		cv::Mat face;//保存人脸

		bool ispre = false;
		int predictedLabel = 0;
		int facenum = 0;
		while (_capture.read(_frame))
		{
			char c = waitKey(33);//控制帧率
			cv::imshow("predect", _frame);//显示视频

			if (c == 'p')//按键p则预测人脸
			{
				ispre = true;
				std::cout << std::endl << "预测此帧视频的人脸" << std::endl;
			}

			if (c == 'q')//按键q退出人脸预测
			{
				std::cout << "停止预测人脸" << std::endl;
				break;
			}

			//if (ispre)//表明用户按下了字母p键，开始检测并识别人脸
			{
				ispre = false;//人脸预测标志位为假，确保每次只预测一次人脸

				_frame.copyTo(temp);//复制当前视频帧				

				if (usepedestrianrects)//如果使用行人检测的矩形框来检测人脸
				{
					facedect(temp,1);//在当前帧中检测人脸
					for (std::vector<std::vector<std::vector<cv::Rect>>>::iterator itallpedfacerects = _allpedestrianfacerects.begin(); itallpedfacerects != _allpedestrianfacerects.end(); ++itallpedfacerects)
					{
						for (std::vector<std::vector<cv::Rect>>::iterator itpedfacerects = itallpedfacerects->begin(); itpedfacerects != itallpedfacerects->end(); ++itpedfacerects)
						{
							for (std::vector<cv::Rect>::iterator itrects = itpedfacerects->begin(); itrects != itpedfacerects->end(); ++itrects)
							{
								face = cv::Mat(_frame, *itrects);
								resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像，确保与模型训练时采取的图片大小相同
								cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
								toGrayscale(face);//归一化	
								predictedLabel = _model->predict(face);
								++facenum;
								cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//将检测到的人脸显示出来
								std::cout << "检测到是" << predictedLabel << std::endl;
							}
						}
					}
				}
				else//
				{
					int facenum = facedect(temp);//在当前帧中检测人脸
					for (std::vector<std::vector<cv::Rect>>::iterator itallrects = _allfacerects.begin(); itallrects != _allfacerects.end(); ++itallrects)
					{
						for (std::vector<cv::Rect>::iterator itrects = itallrects->begin(); itrects != itallrects->end(); ++itrects)
						{
							face = cv::Mat(_frame, *itrects);
							resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像，确保与模型训练时采取的图片大小相同
							cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
							toGrayscale(face);//归一化	
							predictedLabel = _model->predict(face);
							++facenum;
							cv::imshow(format("%d_%d_.jpg", facenum, predictedLabel), face);//将检测到的人脸显示出来
							std::cout << "检测到是" << predictedLabel << std::endl;
						}
					}
				}
			}
			if (savevideobool)
			{
				_capsave << _frame;//保存视频帧
			}
		}
	}
}

void faceclass::showeigenface(bool eigenface)
{
	if (_facemodelno == 1 && eigenface)
	{
		//被注释起来的是PCA模型的参数
		// 有时候你想要获得或设置模型内部的数据，但是在cv::FaceRecognizer中却没有办法获得
		// 由于cv::FaceRecognizer是由cv::Algorithm派生而来，你可以从cv::Algorithm中获取数据。
		// 首先，在没有重新训练模型的情况下，设置FaceRecognizer的阈值为0.0。这将对模型评估很有效。
		//model->set("threshold", 0.0);//没看懂有什么用
		// 现在模型的阈值为0.0。由于不可能在它之下有一个距离，现在预测将会返回-1
		//predictedLabel = model->predict(face);
		//cout << "预测的类是 = " << predictedLabel << endl;
		// 下面是如何获得特征脸模型的特征值：
		Mat eigenvalues = _model->getMat("eigenvalues");
		// 同样的我们可以读取特征脸来获得特征向量：
		Mat W = _model->getMat("eigenvectors");
		// 显示前10个特征脸:
		for (int i = 0; i < min(10, W.cols); i++)
		{
			string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
			cout << msg << endl;
			// 取得特征向量 #i
			Mat ev = W.col(i).clone();
			// 变化为原来的图片大小并且归一化到[0...255]用来显示
			Mat grayscale = toGrayscale(ev.reshape(1, 200));//Mat grayscale = toGrayscale(ev.reshape(1, height));
			// 显示图片并且运用彩色图片获得更好的效果。
			Mat cgrayscale;
			cv::applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
			imshow(format("%d_.jpg", i), cgrayscale);
			imshow(format("%d.jpg", i), grayscale);
			imwrite(format("%d_.jpg", i), cgrayscale);
			imwrite(format("%d.jpg", i), grayscale);
		}
	}
	else
	{
		std::cout << "该模型无法显示特征脸！" << std::endl;
	}
}

bool faceclass::savevideoinit()//初始化保存视频功能
{
	_capture >> _frame;//读取一帧的视频文件
	_capsave.open("监控.avi", CV_FOURCC('M', 'J', 'P', 'G'), 33, _frame.size(), 1);
	_capsave << _frame;//保存一帧的视频文件
	if (!_capsave.isOpened())//判断保存视频是否正确初始化
	{
		cout << "保存视频失败!" << endl;
		return false;
	}
	else
	{
		std::cout << "保存视频函数初始化成功！" << std::endl;
		return true;
	}
}


