#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H
#include <time.h>
#include "base.h"

class ORBextractor
{
public:
	enum{HARRIS_SCORE=0,FAST_SCORE=1};
	ORBextractor(int _nFeatures, float _scaleFactor,int _nlevels)
	{
		nFeatures=_nFeatures;
		scaleFactor=_scaleFactor;
		nlevels=_nlevels;
	}

	void extract(cv::Mat &img, vector<cv::KeyPoint> &mvKeypoints, cv::Mat &mDescritpors)
	{
		
		//Timer timer;
		
		//timer.
		//extractor
		cv::ORB orb(nFeatures,scaleFactor,nlevels);
		orb(img,cv::Mat(), mvKeypoints, mDescritpors, false);

 		//cout<<"ORB keypoint number: "<<mvKeypoints.size()<<endl;
		return ;
	}
	~ORBextractor(){}
	
private:
	int 	nFeatures;
	float 	scaleFactor;
	int 	nlevels;


};

#endif
