#pragma comment (lib,"ws2_32.lib")
#include <Winsock2.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>//遍历文件图片的时候需要的头文件
//#include<conio.h>

#include "sample.h"
#include "faceclass.h"

using namespace cv;
using namespace std;

int main()
{
	//版本协商
	WORD wVersionRequested;
	WSADATA wsaData;
	int err;

	wVersionRequested = MAKEWORD(1, 1); //0x0101
	err = WSAStartup(wVersionRequested, &wsaData);

	if (err != 0)
	{
		return 0;
	}

	if (LOBYTE(wsaData.wVersion) != 1 || HIBYTE(wsaData.wVersion) != 1)    //wsaData.wVersion!=0x0101
	{
		WSACleanup();
		return 0;
	}

	//创建连向服务器的套接字
	SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);

	//创建地址信息
	SOCKADDR_IN hostAddr;
	//hostAddr.sin_addr.S_un.S_addr = inet_addr("192.168.1.155");
	hostAddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	hostAddr.sin_family = AF_INET;
	hostAddr.sin_port = htons(6000);

	sample kylsample;
	kylsample.addcascade();
	//kylsample.runvedio("nothing");
	//kylsample.takephoto("11", "Ann", "1");//diaoyong

	faceclass kylface;
	kylface.addcascade();
	kylface.setmodelno();
	kylface.loadfacemodel();
	kylface.opencamera();
	//kylface.runvedio("nothing");//diaoyong

	////kylface.train_new_model();//diaoyong
	//
	//kylface.smartdect(true,true);//diaoyong
	//kylface.userdect(true,true);//diaoyong

	int flag = 3;

	string sample_label = "11", sample_name = "Ann", sample_num = "1";


	while (true)
	{
		//连接服务器
		connect(sock, (sockaddr*)&hostAddr, sizeof(sockaddr));
		char revBuf[128];
		//从服务器获得数据
		recv(sock, revBuf, 128, 0);
		printf("%s TcpServer\n", revBuf);
		//向服务器发送数据
		//send(sock,"1",12,0);
		closesocket(sock);
		switch (*revBuf){
		case '1':
			flag = 1;
			break;
		case '2':
			flag = 2;
			break;
		case '3':
			flag = 3;
			break;
		case '4':
			flag = 4;
			break;
		case '5':
			flag = 5;
			break;
		}
		kylface._func = flag;
		kylsample._func = flag;

		kylface.runvedio("nothing", flag);//diaoyong
		kylface.smartdect(flag, true, false);//diaoyong
		kylface.userdect(flag, true, false);//diaoyong

		if (flag == 4){
			kylface.train_new_model();//diaoyong
			flag = -1;//
		}

		if (flag == 5){
			kylsample.runvedio("nothing");
			kylsample.takephoto(flag, sample_label, sample_name, sample_num);//diaoyong
			flag = -1;
		}

	}


	return 0;
}






