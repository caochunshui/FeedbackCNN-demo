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
//* im: ����ԭͼƬ
//* salim��saliency map
//* outim������Ľ��ͼƬ
//* intermediate���Ƿ�����м���ͼ
//* intermediatePath:�м���ͼ��·��
//* intermediatePrefix:�м���ͼ��ǰ׺
//* thres����ʼ��ʱ���ֵ������ֵ
//* maxIter��grabcut��������
//* borderWidth��grabcut��mask�ı����߿���*/
//************************************************************************//
void saliencyCut(Mat im, Mat salim, Mat &outim, bool intermediate = true, string intermediatePath = "./", string intermediatePrefix = "", int thres = 70, int maxIter = 4, int borderWidth = 15);

