//

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>
#include "sample.h"

extern void takephoto(string labelcin, string sample_name, string sample_no);

using namespace cv;
using namespace std;


void sample::help()
{
	cout << "按键‘b’：开始采集人脸；\n按键‘t’：暂停/开始采集人脸；\n按键‘f’：此次人脸采集结束，若再按键‘b’开始下一次的人脸采集；\n按键‘q’：退出采集人脸函数，表示所有人脸样本采集完毕" << endl;
	cout << "输入这些字母时需要在视频窗口上；输入标号和序号时在交互窗口上" << endl;
}

//sample.h头文件中有cascade_name的默认值
bool sample::addcascade(char* cascade_name)
{
	_cascade_name = cascade_name;

	if (!_frontalface_cascade.load(_cascade_name))//判断Haar特征加载是否成功
	{
		std::cout << "无法加载级联分类器文件！" << std::endl;
		return false;
	}
	else
	{
		std::cout << "成功加载级联分类器" << cascade_name << std::endl;
		return true;
	}
}

Mat sample::facedect(cv::Mat image)//检测一张图片里的人脸
{
	vector<Rect> faces;
	Mat face;
	_frontalface_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (!faces.empty()){
		Mat tempface(image, faces[0]);
		_face_rect = faces[0];//将检测到人脸保存到成员值
		tempface.copyTo(face);
		cv::rectangle(image, _face_rect, Scalar(255, 0, 255), 3, 8, 0);
		//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
	}
	return face;
}

//默认打开摄像头，若要打开文件需要将第一个参数设为-2，并将文件名传递给第二个参数
bool sample::opencamera(string filename, int cameranum){
	if (cameranum == -2 && filename != "nothing"){
		_capture.open(filename);
	}
	else if (cameranum != -2 && filename == "nothing"){
		_capture.open(cameranum);
	}
	if (!_capture.isOpened()){
		std::cout << "打开视频文件失败！" << std::endl;
		return false;
	}
	else{
		std::cout << "视频文件已打开！" << std::endl;
		return true;
	}
}

void sample::runvedio(string filename){
	if (opencamera(filename)){
		while (_capture.read(_frame)){
			char c = waitKey(33);//控制帧率
			cv::imshow("vedio", _frame);//显示视频
		}
	}
}

//按键‘b’，开始采集人脸；按键‘t’，暂停/开始采集人脸；按键‘f’，此次人脸采集结束，，若再按键‘b’开始下一次的人脸采集；按键‘q’，退出采集人脸函数，表示所有人脸样本采集完毕
void sample::takephoto(string labelcin, string sample_name, string sample_no)
{


	bool beginface = false;//是否开始采集人脸的标志位

	//Mat frame;//保存视频帧
	//Mat temp;//采集人脸时临时复制当前视频帧
	Mat face;//保存采集到的人脸
	//namedWindow("img");
	int label = 0;//当前人脸的标号，每采集一次自加1
	//string labelcin;//定义一个接收标号的字符串
	//string facenumcin;//定义一个接收输入人脸序号的字符串
	//string facenum;//定义一个人脸序号的字符串，用以保存图片名称时使用
	int faceno = 0;//存储人脸的标号
	int facetotal = 0;//保存当前采集人脸的个数
	int Allfacetotal = 0;//采集人脸的总数
	long frameNo = 0;//视频的帧数
	long NowframNo = 0;//保存开始采集人脸时的视频帧数

	string name = "f";

	while (_capture.read(_frame))
	{
		frameNo++;//帧数加1
		char c = waitKey(33);//控制帧率
		if (c == 'b')//如果按键‘b’，开始采集人脸
		{
			while (1)
			{
				cout << "请输入待采集人脸的标号：";
				//cin >> labelcin;//输入标号
				int labelnum = 0;//存储输入字符串中数字的个数
				for (unsigned int l = 0; l < labelcin.length(); l++)//循环判断输入的标号字符串是否是数字
				{
					if (!isdigit(labelcin[l]))//使用isdigit函数可以判断每一位是否是0-9的数字，如果输入的字符串有有任何一位不是数字则要重新输入
					{
						cout << endl << "请输入数字！" << endl;
						break;//退出for循环
					}
					else
					{
						labelnum++;//如果判断此次循环的字符串中的是数字，labelnum加1
					}
				}
				if (labelnum == labelcin.length())//如果数字的长度等于输入字符串的长度，表示输入的都是数字
				{
					stringstream ss1;//stringstream可以吞下不同的类型，根据要求的类型，然后吐出不同的类型
					ss1 << labelcin;
					ss1 >> label;//将输入的标号转换为int型，从而可以将标号压栈存储
					break;//退出while循环
				}
			}
			while (1)
			{
				cout << "请输入待采集人脸开始的序号：";
				//cin >> sample_no;//输入标号
				int facenlen = 0;//存储输入字符串中数字的个数
				for (unsigned int f = 0; f < sample_no.length(); f++)//循环判断输入的标号字符串是否是数字
				{
					if (!isdigit(sample_no[f]))//使用isdigit函数可以判断每一位是否是0-9的数字，如果输入的字符串有有任何一位不是数字则要重新输入
					{
						cout << endl << "请输入数字！" << endl;
						break;//退出for循环
					}
					else
					{
						facenlen++;//如果判断此次循环的字符串中的是数字，labelnum加1
					}
				}
				if (facenlen == sample_no.length())//如果数字的长度等于输入字符串的长度，表示输入的都是数字
				{
					stringstream ss2;//stringstream可以吞下不同的类型，根据要求的类型，然后吐出不同的类型
					ss2 << sample_no;
					ss2 >> faceno;//将输入的序号转换为int型，从而可以自加1，以区别人脸的名称
					faceno -= 1;//先将输入的值减去1，因为后面的循环里有加1，使得采集头像的序号从输入的序号开始
					break;//退出while循环
				}
			}
			//如果输入的标号和序号满足要求则进入如下操作
			beginface = true;//将标志位设为真
			name = "f";//每次开始采集人脸都重新给名字幅值
			name += labelcin;//name变为n加上标号，如n1,n2……
			name += "_";//name变为n1_,n2_……
			NowframNo = frameNo;//保存当前的帧数，用以判断当前帧开始后的每30帧
			cout << endl << endl << "开始采集人脸，此人标号为" << label << endl << endl;
		}
		//满足开始检测的条件，从当前时刻开始每30帧采集一次人脸
		if ((frameNo - NowframNo) % 30 == 0)
		{
			//_frame.copyTo(temp);//复制当前视频帧
			face = facedect(_frame);//在当前帧中检测人脸
			if (face.empty())//没有检测到人脸则显示错误
			{
				cout << endl << endl << "没有检测到人脸，请对准摄像头！" << endl << endl;
				//break;//不可以用break，否则会退出样本采集
			}
			else//若有检测到人脸则进行如下操作
			{
				resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像
				cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
				//toGrayscale(face);//归一化
				cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);

				facetotal++;//每进入一次循环人脸数自加1，用来记录此次采集的人脸总数
				Allfacetotal++;//保存总共采集到的人脸的个数
				faceno++;//将输入的序号自加1，以区别人脸的名称
				stringstream ss3;//stringstream可以吞下不同的类型，根据要求的类型，然后吐出不同的类型
				ss3 << faceno;//吞下faceno，转换成string类型（facenum）用以保存图片名称
				ss3 >> sample_no;//将输入的序号转换为int型，从而可以自加1，以区别人脸的名称
				name += sample_no;//name变为n加上标号，如n1_1,n1_2……
				name += ".jpg";

				imwrite(name, face);
				cout << endl << endl << "检测到并保存人脸图片" << name << endl;
				name.pop_back();
				name.pop_back();
				name.pop_back();
				name.pop_back();//以上4次是弹出".jpg"

				int weishu = 0;//定义人脸序号的位数
				int facenocopy = faceno;//将当前人脸序号复制保存
				while (facenocopy)//求出人脸序号的位数
				{
					facenocopy /= 10;//人脸序号依次除以10判断
					weishu++;//每次除以10后将weishu变量加1
				}
				for (int i = 0; i < weishu; i++)//循环弹出标号
				{
					name.pop_back();//弹出标号
				}
				cout << "目前采集到此人脸的个数为" << facetotal << endl << endl;
			}
		}

		imshow("img", _frame);//显示视频帧

	}
	destroyWindow("img");//关闭视频窗口
}
