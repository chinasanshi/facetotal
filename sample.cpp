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
	cout << "������b������ʼ�ɼ�������\n������t������ͣ/��ʼ�ɼ�������\n������f�����˴������ɼ����������ٰ�����b����ʼ��һ�ε������ɼ���\n������q�����˳��ɼ�������������ʾ�������������ɼ����" << endl;
	cout << "������Щ��ĸʱ��Ҫ����Ƶ�����ϣ������ź����ʱ�ڽ���������" << endl;
}

//sample.hͷ�ļ�����cascade_name��Ĭ��ֵ
bool sample::addcascade(char* cascade_name)
{
	_cascade_name = cascade_name;

	if (!_frontalface_cascade.load(_cascade_name))//�ж�Haar���������Ƿ�ɹ�
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

Mat sample::facedect(cv::Mat image)//���һ��ͼƬ�������
{
	vector<Rect> faces;
	Mat face;
	_frontalface_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (!faces.empty()){
		Mat tempface(image, faces[0]);
		_face_rect = faces[0];//����⵽�������浽��Աֵ
		tempface.copyTo(face);
		cv::rectangle(image, _face_rect, Scalar(255, 0, 255), 3, 8, 0);
		//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
	}
	return face;
}

//Ĭ�ϴ�����ͷ����Ҫ���ļ���Ҫ����һ��������Ϊ-2�������ļ������ݸ��ڶ�������
bool sample::opencamera(string filename, int cameranum){
	if (cameranum == -2 && filename != "nothing"){
		_capture.open(filename);
	}
	else if (cameranum != -2 && filename == "nothing"){
		_capture.open(cameranum);
	}
	if (!_capture.isOpened()){
		std::cout << "����Ƶ�ļ�ʧ�ܣ�" << std::endl;
		return false;
	}
	else{
		std::cout << "��Ƶ�ļ��Ѵ򿪣�" << std::endl;
		return true;
	}
}

void sample::runvedio(string filename){
	if (opencamera(filename)){
		while (_capture.read(_frame)){
			char c = waitKey(33);//����֡��
			cv::imshow("vedio", _frame);//��ʾ��Ƶ
		}
	}
}

//������b������ʼ�ɼ�������������t������ͣ/��ʼ�ɼ�������������f�����˴������ɼ������������ٰ�����b����ʼ��һ�ε������ɼ���������q�����˳��ɼ�������������ʾ�������������ɼ����
void sample::takephoto(string labelcin, string sample_name, string sample_no)
{


	bool beginface = false;//�Ƿ�ʼ�ɼ������ı�־λ

	//Mat frame;//������Ƶ֡
	//Mat temp;//�ɼ�����ʱ��ʱ���Ƶ�ǰ��Ƶ֡
	Mat face;//����ɼ���������
	//namedWindow("img");
	int label = 0;//��ǰ�����ı�ţ�ÿ�ɼ�һ���Լ�1
	//string labelcin;//����һ�����ձ�ŵ��ַ���
	//string facenumcin;//����һ����������������ŵ��ַ���
	//string facenum;//����һ��������ŵ��ַ��������Ա���ͼƬ����ʱʹ��
	int faceno = 0;//�洢�����ı��
	int facetotal = 0;//���浱ǰ�ɼ������ĸ���
	int Allfacetotal = 0;//�ɼ�����������
	long frameNo = 0;//��Ƶ��֡��
	long NowframNo = 0;//���濪ʼ�ɼ�����ʱ����Ƶ֡��

	string name = "f";

	while (_capture.read(_frame))
	{
		frameNo++;//֡����1
		char c = waitKey(33);//����֡��
		if (c == 'b')//���������b������ʼ�ɼ�����
		{
			while (1)
			{
				cout << "��������ɼ������ı�ţ�";
				//cin >> labelcin;//������
				int labelnum = 0;//�洢�����ַ��������ֵĸ���
				for (unsigned int l = 0; l < labelcin.length(); l++)//ѭ���ж�����ı���ַ����Ƿ�������
				{
					if (!isdigit(labelcin[l]))//ʹ��isdigit���������ж�ÿһλ�Ƿ���0-9�����֣����������ַ��������κ�һλ����������Ҫ��������
					{
						cout << endl << "���������֣�" << endl;
						break;//�˳�forѭ��
					}
					else
					{
						labelnum++;//����жϴ˴�ѭ�����ַ����е������֣�labelnum��1
					}
				}
				if (labelnum == labelcin.length())//������ֵĳ��ȵ��������ַ����ĳ��ȣ���ʾ����Ķ�������
				{
					stringstream ss1;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
					ss1 << labelcin;
					ss1 >> label;//������ı��ת��Ϊint�ͣ��Ӷ����Խ����ѹջ�洢
					break;//�˳�whileѭ��
				}
			}
			while (1)
			{
				cout << "��������ɼ�������ʼ����ţ�";
				//cin >> sample_no;//������
				int facenlen = 0;//�洢�����ַ��������ֵĸ���
				for (unsigned int f = 0; f < sample_no.length(); f++)//ѭ���ж�����ı���ַ����Ƿ�������
				{
					if (!isdigit(sample_no[f]))//ʹ��isdigit���������ж�ÿһλ�Ƿ���0-9�����֣����������ַ��������κ�һλ����������Ҫ��������
					{
						cout << endl << "���������֣�" << endl;
						break;//�˳�forѭ��
					}
					else
					{
						facenlen++;//����жϴ˴�ѭ�����ַ����е������֣�labelnum��1
					}
				}
				if (facenlen == sample_no.length())//������ֵĳ��ȵ��������ַ����ĳ��ȣ���ʾ����Ķ�������
				{
					stringstream ss2;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
					ss2 << sample_no;
					ss2 >> faceno;//����������ת��Ϊint�ͣ��Ӷ������Լ�1������������������
					faceno -= 1;//�Ƚ������ֵ��ȥ1����Ϊ�����ѭ�����м�1��ʹ�òɼ�ͷ�����Ŵ��������ſ�ʼ
					break;//�˳�whileѭ��
				}
			}
			//�������ı�ź��������Ҫ����������²���
			beginface = true;//����־λ��Ϊ��
			name = "f";//ÿ�ο�ʼ�ɼ����������¸����ַ�ֵ
			name += labelcin;//name��Ϊn���ϱ�ţ���n1,n2����
			name += "_";//name��Ϊn1_,n2_����
			NowframNo = frameNo;//���浱ǰ��֡���������жϵ�ǰ֡��ʼ���ÿ30֡
			cout << endl << endl << "��ʼ�ɼ����������˱��Ϊ" << label << endl << endl;
		}
		//���㿪ʼ�����������ӵ�ǰʱ�̿�ʼÿ30֡�ɼ�һ������
		if ((frameNo - NowframNo) % 30 == 0)
		{
			//_frame.copyTo(temp);//���Ƶ�ǰ��Ƶ֡
			face = facedect(_frame);//�ڵ�ǰ֡�м������
			if (face.empty())//û�м�⵽��������ʾ����
			{
				cout << endl << endl << "û�м�⵽���������׼����ͷ��" << endl << endl;
				//break;//��������break��������˳������ɼ�
			}
			else//���м�⵽������������²���
			{
				resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��
				cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
				//toGrayscale(face);//��һ��
				cv::normalize(face, face, 0, 255, NORM_MINMAX, CV_8UC1);

				facetotal++;//ÿ����һ��ѭ���������Լ�1��������¼�˴βɼ�����������
				Allfacetotal++;//�����ܹ��ɼ����������ĸ���
				faceno++;//�����������Լ�1������������������
				stringstream ss3;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
				ss3 << faceno;//����faceno��ת����string���ͣ�facenum�����Ա���ͼƬ����
				ss3 >> sample_no;//����������ת��Ϊint�ͣ��Ӷ������Լ�1������������������
				name += sample_no;//name��Ϊn���ϱ�ţ���n1_1,n1_2����
				name += ".jpg";

				imwrite(name, face);
				cout << endl << endl << "��⵽����������ͼƬ" << name << endl;
				name.pop_back();
				name.pop_back();
				name.pop_back();
				name.pop_back();//����4���ǵ���".jpg"

				int weishu = 0;//����������ŵ�λ��
				int facenocopy = faceno;//����ǰ������Ÿ��Ʊ���
				while (facenocopy)//���������ŵ�λ��
				{
					facenocopy /= 10;//����������γ���10�ж�
					weishu++;//ÿ�γ���10��weishu������1
				}
				for (int i = 0; i < weishu; i++)//ѭ���������
				{
					name.pop_back();//�������
				}
				cout << "Ŀǰ�ɼ����������ĸ���Ϊ" << facetotal << endl << endl;
			}
		}

		imshow("img", _frame);//��ʾ��Ƶ֡

	}
	destroyWindow("img");//�ر���Ƶ����
}
