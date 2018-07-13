#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/util/SaliencyCut.h"



using namespace std;
using namespace cv;

int main(){
	Mat im, salim, outim;
	im = imread("matlab/new_attention/show_forgrabcut/ori6.jpg");
        std::cout<<im.size().width<<" "<<im.size().height<<std::endl;
	salim = imread("matlab/new_attention/show_forgrabcut/sal_6.jpg", 0);
        salim=255*salim;
	saliencyCut(im, salim, outim, true);
	//imshow("result", outim);
	imwrite("45.out.png", outim);
	//waitKey(0);
	return 0;
}
