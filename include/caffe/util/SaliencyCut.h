#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>

//#include "cv.h"
//#include "highgui.h"
#include "caffe/common.hpp"
using namespace std;
using namespace cv;

//************************************************************************/
//* im: 输入原图片
//* salim：saliency map
//* outim：输出的结果图片
//* intermediate：是否输出中间结果图
//* intermediatePath:中间结果图的路径
//* intermediatePrefix:中间结果图的前缀
//* thres：初始化时候二值化的阈值
//* maxIter：grabcut迭代次数
//* borderWidth：grabcut中mask的背景边框宽度*/
//************************************************************************//
void saliencyCut(Mat im, Mat salim, Mat &outim, bool intermediate = true, string intermediatePath = "./", string intermediatePrefix = "", int thres = 70, int maxIter = 4, int borderWidth = 15);

