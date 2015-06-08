//

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>
#include "captureface.h"

using namespace cv;
using namespace std;


void captureface::help()
{
	cout << "������b������ʼ�ɼ�������\n������t������ͣ/��ʼ�ɼ�������\n������f�����˴������ɼ����������ٰ�����b����ʼ��һ�ε������ɼ���\n������q�����˳��ɼ�������������ʾ�������������ɼ����" << endl;
	cout << "������Щ��ĸʱ��Ҫ����Ƶ�����ϣ������ź����ʱ�ڽ���������" << endl;
}

Mat captureface::facedect(cv::Mat image)//���һ��ͼƬ�������
{
	char* cascade_name = //"C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
		"haarcascade_frontalface_alt.xml";
	CascadeClassifier frontalface_cascade;
	if (!frontalface_cascade.load(cascade_name))//�ж�Haar���������Ƿ�ɹ�
	{
		printf("�޷����ؼ����������ļ���\n");
	}
	vector<Rect> faces;
	Mat face;
	frontalface_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (!faces.empty())
	{
		Mat tempface(image, faces[0]);
		tempface.copyTo(face);
	}
	return face;
}
//ֻ����Ϊ�Ҷ�ͼ�����򱨴�Ȼ�󽫻Ҷ�ͼ��һ��
cv::Mat captureface::toGrayscale(cv::Mat src)
{
	// ֻ����Ϊ��ͨ����
	if (src.channels() != 1)
	{
		CV_Error(CV_StsBadArg, "ֻ֧�ֵ�ͨ���ľ���");
	}
	// ���������ع�һ�����ͼƬ
	Mat dst;
	cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

//������b������ʼ�ɼ�������������t������ͣ/��ʼ�ɼ�������������f�����˴������ɼ������������ٰ�����b����ʼ��һ�ε������ɼ���������q�����˳��ɼ�������������ʾ�������������ɼ����
void captureface::takphoto()
{
	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "������ͷʧ�ܣ�" << endl;
	}

	bool beginface = false;//�Ƿ�ʼ�ɼ������ı�־λ

	Mat frame;//������Ƶ֡
	Mat temp;//�ɼ�����ʱ��ʱ���Ƶ�ǰ��Ƶ֡
	Mat face;//����ɼ���������
	//Mat not;//���水λȡ�������Ƶͼ��
	namedWindow("img");
	int label = 0;//��ǰ�����ı�ţ�ÿ�ɼ�һ���Լ�1
	string labelcin;//����һ�����ձ�ŵ��ַ���
	string facenumcin;//����һ����������������ŵ��ַ���
	string facenum;//����һ��������ŵ��ַ��������Ա���ͼƬ����ʱʹ��
	int faceno = 0;//�洢�����ı��
	int facetotal = 0;//���浱ǰ�ɼ������ĸ���
	int Allfacetotal = 0;//�ɼ�����������
	long frameNo = 0;//��Ƶ��֡��
	long NowframNo = 0;//���濪ʼ�ɼ�����ʱ����Ƶ֡��

	string name = "f";

	while (capture.read(frame))
	{
		frameNo++;//֡����1
		//cout << "��ǰ֡��Ϊ" << frameNo << "; "; //�����ǰ֡��
		//bitwise_not(frame,not);//��frame��λȡ���õ�ȡ�����ͼ��
		//imshow("not", not);//��ʾ��λȡ�����ͼ��
		imshow("img", frame);//��ʾ��Ƶ֡
		char c = waitKey(33);//����֡��
		if (c == 'b')//���������b������ʼ�ɼ�����
		{
			while (1)
			{
				cout << "��������ɼ������ı�ţ�";
				cin >> labelcin;//������
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
				cin >> facenumcin;//������
				int facenlen = 0;//�洢�����ַ��������ֵĸ���
				for (unsigned int f = 0; f < facenumcin.length(); f++)//ѭ���ж�����ı���ַ����Ƿ�������
				{
					if (!isdigit(facenumcin[f]))//ʹ��isdigit���������ж�ÿһλ�Ƿ���0-9�����֣����������ַ��������κ�һλ����������Ҫ��������
					{
						cout << endl << "���������֣�" << endl;
						break;//�˳�forѭ��
					}
					else
					{
						facenlen++;//����жϴ˴�ѭ�����ַ����е������֣�labelnum��1
					}
				}
				if (facenlen == facenumcin.length())//������ֵĳ��ȵ��������ַ����ĳ��ȣ���ʾ����Ķ�������
				{
					stringstream ss2;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
					ss2 << facenumcin;
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
		if (((beginface) && (frameNo - NowframNo) % 30 == 0))
		{
			frame.copyTo(temp);//���Ƶ�ǰ��Ƶ֡
			face = facedect(temp);//�ڵ�ǰ֡�м������
			if (face.empty())//û�м�⵽��������ʾ����
			{
				cout << endl << endl << "û�м�⵽���������׼����ͷ��" << endl << endl;
				//break;//��������break��������˳������ɼ�
			}
			else//���м�⵽������������²���
			{
				resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��
				cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
				toGrayscale(face);//��һ��
				//faces.push_back(face);//������ѹջ
				//labels.push_back(label);//�����ѹջ
				facetotal++;//ÿ����һ��ѭ���������Լ�1��������¼�˴βɼ�����������
				Allfacetotal++;//�����ܹ��ɼ����������ĸ���
				faceno++;//�����������Լ�1������������������
				stringstream ss3;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
				ss3 << faceno;//����faceno��ת����string���ͣ�facenum�����Ա���ͼƬ����
				ss3 >> facenum;//����������ת��Ϊint�ͣ��Ӷ������Լ�1������������������
				name += facenum;//name��Ϊn���ϱ�ţ���n1_1,n1_2����	
				name += ".jpg";
				namedWindow("face");
				imshow("face", face);//��ʾ�˴βɼ���������			
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
		if (c == 'f')//���������f�������������ɼ�����
		{
			beginface = false;//��ֵ��һ�βɼ�			
			cout << "�ɼ���������ϣ��˴��ܹ��ɼ�����������Ϊ" << facetotal << endl;
			facetotal = 0;//�����������㣬Ϊ��һλ�����ɼ���׼�����˾����������������ĺ��棬����������ܲɼ��������ͱ�������
			//system("del name");
			//facetotal -= 1;
			//Allfacetotal -= 1;			
			//faces.pop_back();
			//labels.pop_back();//�������һ�βɼ�����������ΪЧ������
			//cout << "�ɼ���������ϣ��������һ�βɼ���ͷ���ܹ��ɼ�����������Ϊ" << facetotal << endl;
		}
		if (c == 'q')//������q�����˳��ɼ�������������ʾ�������������ɼ����
		{
			cout << "���������ɼ���ϣ�" << endl;
			cout << "�ܹ��ɼ�������������Ϊ" << Allfacetotal << endl;
			break;
		}
		if (c == 't')//������t������ͣ�����ɼ����ٰ������ɼ�
		{
			beginface = !beginface;//������־λȡ��
			if (beginface == false)
			{
				cout << "��ͣ�ɼ�������" << endl;//�����ʾ��ʾ��ʱ��ͣ�����ɼ�
			}
			if (beginface == true)
			{
				cout << "�����ɼ�������" << endl;//�����ʾ��ʾ��ʱ���������ɼ�
			}
		}
	}
	destroyWindow("img");//�ر���Ƶ����
	destroyWindow("face");//�ر�face����
}
