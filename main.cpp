//=== this is the main part of this traffic sign detection and recognition project
 //== i finish both detection and recognition function in this .cpp file
 //== struct of this project
 	//=== svm model file is in the ..\svmmodel file
 	//=== traffic video is in the ..\Images file
 	//=== libsvm is in the ..\libsvm file


//== Jinzheng Cai, 2014/07/02

#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "..\libsvm\svm.h"

#define HEIGHT	810
#define WIDTH		1440
#define TABLESIZE	64
#define STEP 2

struct point{ uint8_t h; uint8_t s; uint8_t v; };
struct TrafficRect{ cv::Rect boundRect; bool is_red;};

point table[TABLESIZE][TABLESIZE][TABLESIZE];
const char *IS_SIGN = "..\\svmmodel\\GMT_B_1.model";
const char *WHICH_RED = "..\\svmmodel\\SubBJTRed_B_1.model";
const char *WHICH_BLUE = "..\\svmmodel\\SubBJTBlue_B_1.model";
const std::string LABEL[11] = {"here","lim4M","lim40","lim80","noCar","noPark","noTruck","bike","bikeHere","car","unknown"};


int main( int argc, char** argv )
{
	struct svm_model *is_sign; struct svm_model *which_red; struct svm_model *which_blue;
	cv::Mat frame(HEIGHT,WIDTH,CV_8UC3); cv::Mat hsv_thr_r(HEIGHT,WIDTH,CV_8UC1); cv::Mat hsv_thr_b(HEIGHT,WIDTH,CV_8UC1);
	uchar* pr; uchar* pb; uchar* p2; uint8_t r,g,b;
	cv::vector<cv::vector<cv::Point> > contours; cv::vector<cv::Vec4i> hierarchy; cv::vector<cv::vector<cv::Point> > contours_poly;
	cv::HOGDescriptor *hog = new cv::HOGDescriptor(cvSize(64,64), cvSize(16,16), cvSize(8,8), cvSize(8,8), 9);
	cv::vector<float> desc;
	struct svm_node *svmVec; svmVec = (struct svm_node *)malloc(1765*sizeof(struct svm_node));
	double range;

	//====for write video
	//cv::VideoWriter writer("videoResult.avi",CV_FOURCC('M','J','P','G'),29.0,cv::Size(WIDTH,HEIGHT));

	// create hsv looking up table
	{
		double Min, Delta, S, H, V;
		for(int i=0;i<TABLESIZE;i++) // ref: opencv BGR2HSV, http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html
			for(int j=0;j<TABLESIZE;j++)
				for(int k=0;k<TABLESIZE;k++)
					{
						r = i<<STEP; g = j<<STEP; b=k<<STEP;
						V = cv::max(cv::max(r, g), b); Min = cv::min(cv::min(r, g), b);
						Delta = V - Min;
						S = V == 0 ? 0 : 255*Delta/V;
						if(V==r) H = 60*(g-b)/Delta;
						if(V==g) H = 120+60*(b-r)/Delta;
						if(V==b) H = 240+60*(r-g)/Delta;
						if(H<0)  H += 360;
						table[i][j][k].h = (uint8_t)H; table[i][j][k].s = (uint8_t)S; table[i][j][k].v = (uint8_t)V;
					}
	}

	//=== load svm model
	if((is_sign=svm_load_model(IS_SIGN))==0){fprintf(stderr,"Can't read %s.\n","GMT_B_1.model"); return -1;}
	if((which_red=svm_load_model(WHICH_RED))==0){fprintf(stderr,"Can't read %s.\n","SubBJTRed_B_1.model"); return -1;}
	if((which_blue=svm_load_model(WHICH_BLUE))==0){fprintf(stderr,"Can't read %s.\n","SubBJTBlue_B_1.model"); return -1;}

	//=== load video
	cv::VideoCapture capture("../Images/video.wmv");	//cv::VideoCapture capture("..//Images//IMG_0150.MOV");
	if( !capture.isOpened() ) {fprintf(stderr,"Error when reading video.wmv.\n"); return -1;}
	cv::namedWindow("PlayWMV", CV_WINDOW_KEEPRATIO);

	int64 time= cv::getTickCount();
	//for(int numloop=0;;numloop++) 
	while(1)
		{
			/*if(numloop%200==0)
			{
				printf("finish %d/4000+ frames\n",numloop);
				std::cout<<(cv::getTickCount() - time)/cv::getTickFrequency()<<std::endl; time = cv::getTickCount();
			}*/
			cv::vector<TrafficRect> boundRectS;// time = cv::getTickCount();
			capture >> frame;
			if(frame.empty()) break;
			cv::resize(frame,frame,cv::Size(WIDTH,HEIGHT)); //std::cout<<"Load Image:"<<(cv::getTickCount() - time)/cv::getTickFrequency()<<std::endl; time = cv::getTickCount();
			
			//===looking up table
			for(int i=0;i<HEIGHT;++i)
				{
					//p1 = hsv.ptr<uchar>(i); 
					p2 = frame.ptr<uchar>(i);//获取行指针
					pr = hsv_thr_r.ptr<uchar>(i);  pb = hsv_thr_b.ptr<uchar>(i); 
					for (int j=0;j<WIDTH*3;j+=3)
					{
						r = p2[j]>>STEP; g = p2[j+1]>>STEP; b = p2[j+2]>>STEP; 			
						point temp=table[r][g][b];
						//p1[j]=table[r][g][b].h; p1[j+1]=table[r][g][b].s; p1[j+2]=table[r][g][b].v;
						//h=table[r][g][b][1]; s=table[r][g][b][2]; v=table[r][g][b][3];
						*pr = 255*(((temp.h>210)&(temp.h<255)&(temp.s>50)&(temp.s<160)&(temp.v>50)&(temp.v<245))
							|((temp.h>60)&(temp.h<80)&(temp.s>30)&(temp.s<70)&(temp.v>50)&(temp.v<80))); *pr++;
						*pb = 255*((temp.h>20)&(temp.h<40)&(temp.s>50)&(temp.s<215)&(temp.v>80)&(temp.v<140)); *pb++;	// not safe but high speed
					}
				}	
			cv::blur(hsv_thr_r,hsv_thr_r,cv::Size(3,3)); cv::blur(hsv_thr_b,hsv_thr_b,cv::Size(3,3)); // combine small crack

			//===
			//std::cout<<"RGB2HSV and Threshold Color:"<<(cv::getTickCount() - time)/cv::getTickFrequency()<<std::endl; time = cv::getTickCount();
			// detect red area, configure the contours
			cv::findContours(hsv_thr_r,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
			contours_poly.resize(contours.size());
			for(int i=0;i<(int)contours.size();i++)
				{
					cv::approxPolyDP(cv::Mat(contours[i]),contours_poly[i],3,true);
					cv::Rect boundRect = cv::boundingRect(cv::Mat(contours_poly[i])); double width = boundRect.width; double height = boundRect.height;
					if ((width>=30)&(height>=30)&(width<=150)&(height<=150)&(width/height<=1.3)&(height/width<=1.3))
						{ TrafficRect temp; temp.boundRect = boundRect; temp.is_red = 1; boundRectS.push_back(temp);}
				}
			// detect blue area, configure the contours
			cv::findContours(hsv_thr_b,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
			contours_poly.resize(contours.size());
			for(int i=0;i<(int)contours.size();i++)
				{
					cv::approxPolyDP(cv::Mat(contours[i]),contours_poly[i],3,true);
					cv::Rect boundRect = cv::boundingRect(cv::Mat(contours_poly[i])); double width = boundRect.width; double height = boundRect.height;
					if ((width>=15)&(height>=15)&(width<=130)&(height<=130)&(width/height<=1.3)&(height/width<=1.7))
						{ TrafficRect temp; temp.boundRect = boundRect; temp.is_red = 0; boundRectS.push_back(temp);}
				}
			//std::cout<<"Detect Countours:"<<(cv::getTickCount() - time)/cv::getTickFrequency()<<std::endl; time = cv::getTickCount();
			//get ROI patches, extract Hog, prediction
			for(int i=0;i<(int)boundRectS.size();i++)
				{
					cv::Mat tempImg; frame(boundRectS[i].boundRect).copyTo(tempImg);
					cv::resize(tempImg,tempImg,cv::Size(64,64));
					hog->compute(tempImg,desc);
					auto result = std::minmax_element(desc.begin(),desc.end());
					range = *result.second - *result.first;
					for(int k=0; k<(int)desc.size(); k++)
						{
								svmVec[k].index=k+1;
								svmVec[k].value=(desc[k]-*result.first)/range;
						}
					svmVec[1764].index=-1;
					if(svm_predict(is_sign,svmVec)==1)
					{
						int label=10;
						switch (boundRectS[i].is_red)
						{
							case 1:  label = (int)svm_predict(which_red,svmVec); break;
							case 0:  label = (int)svm_predict(which_blue,svmVec); break;
						}
						rectangle( frame, boundRectS[i].boundRect.tl(), boundRectS[i].boundRect.br(), cv::Scalar(255,0,0), 2, 8, 0 );
						cv::putText( frame, LABEL[label], boundRectS[i].boundRect.br(), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,0,0),3,8);
					}
				}
			//std::cout<<"Prediction:" <<(cv::getTickCount() - time)/cv::getTickFrequency()<<std::endl; time = cv::getTickCount();
			cv::imshow("PlayWMV",frame); cv::waitKey(1);
			//writer << frame;
			//std::cout<<"Display:" <<(cv::getTickCount() - time)/cv::getTickFrequency()<<std::endl;
			//std::cout<<"\n"<<std::endl;
	}
	//std::cout<<(cv::getTickCount() - time)/cv::getTickFrequency()<<std::endl; time = cv::getTickCount();
	//writer.release();
	capture.release();
	cvDestroyWindow("PlayWMV"); 

	return 0;
}
