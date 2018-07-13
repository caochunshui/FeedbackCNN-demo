#include "caffe/util/SaliencyCut.h"

void setMaskBorder(Mat &mask, int borderWidth = 15){
	Rect rect(borderWidth, borderWidth, mask.size().width-borderWidth*2, mask.size().height-borderWidth*2);
        std::cout<<borderWidth<<" "<<mask.size().width-borderWidth*2<<" "<<mask.size().height-borderWidth*2<<std::endl;
	Mat tmp(mask.size(), CV_8UC1, 1);
        Mat zero_tmp=tmp(rect);
        zero_tmp=0*zero_tmp;
	tmp(rect) = zero_tmp;
	mask.setTo(GC_BGD, tmp);
	//imshow("roi", mask);
}

void saliencyCut(Mat im, Mat mask, Mat &outim, bool intermediate, string intermediatePath, string intermediatePrefix, int thres, int maxIter, int borderWidth){
	//salim.copyTo(outim);
	//����thres����ֵ��ͼ��
      //	threshold(salim, outim, thres, 255, CV_THRESH_BINARY);
	
	//��ʼ��mask
//	Mat mask(im.size(), CV_8UC1, Scalar(GC_PR_FGD));
//	mask.setTo(GC_PR_BGD, outim==0);
//	setMaskBorder(mask, borderWidth);
	//imshow("mask", mask);
	//waitKey(0);
	//��һ��grabcut
	Mat bgdmodel, fgdmodel;
	Rect rect;//(10, 10, im.size().width-20, im.size().height-20);//ʵ����rect�ڴ�������û�����ϵ�
	if (sum(mask&1).val[0] > 0){
		grabCut(im, mask, rect, bgdmodel, fgdmodel, 1, GC_INIT_WITH_MASK);
	}

	outim.setTo(0);
	outim.setTo(255, mask&1);

	//���������flag����ô����м���
	if (intermediate){
		char outimname[20];
//		imwrite(intermediatePath+intermediatePrefix+"iter0.jpg", outim);
	}
        
	//dilation��erosion��kernel�Ķ���
	Mat dilation_dst;
	int dilation_type = MORPH_RECT;
	int dilation_size = 5;
	Mat d_element = getStructuringElement( dilation_type,
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );
	Mat erosion_dst;
	int erosion_type = MORPH_RECT;
	int erosion_size = 5;
	Mat e_element = getStructuringElement( erosion_type,
		Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		Point( erosion_size, erosion_size ) );
	
	//���е���
	int iter = 0;
	while (iter++<maxIter){ 
                
		//dilation
		dilate( outim, dilation_dst, d_element );
		//imshow( "Dilation Demo", dilation_dst );

		//erosion	
		erode( outim, erosion_dst, e_element );
		//imshow( "Erosion Demo", erosion_dst );

		//����grabcut�µ�mask
		mask.setTo(GC_PR_FGD);//������������Ϊ���ܵ�ǰ��
		mask.setTo(GC_BGD, dilation_dst<128);//���ͺ�ı���������mask�ı���
		mask.setTo(GC_FGD, erosion_dst >= 128);//��ʴ���ǰ����mask��ǰ��
		//setMaskBorder(mask, borderWidth);//���ñ����߿�
		//imshow("mask", mask);

		//ʹ��grabcut(ע�⵽Խ���ж�)
		if (sum(mask&1).val[0] > 0){
			grabCut(im, mask, rect, bgdmodel, fgdmodel, 1, GC_INIT_WITH_MASK);
		}
		

		//�������ͼƬ
		outim.setTo(0);
		outim.setTo(255, mask&1);

		//����м���
		if (intermediate){
			char outimname[20];
			sprintf(outimname, "iter%d.jpg", iter);
//			imwrite(intermediatePath+intermediatePrefix+outimname, outim);
		}
	}

}


