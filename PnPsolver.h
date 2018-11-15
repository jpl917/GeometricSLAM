#ifndef _PNPSOLVER_H_
#define _PNPSOLVER_H_
#include "base.h"
#include "frame.h"
#include "utils.h"

class PnPsolver
{
public:
	//void calculateCorrespondence();
	vector<DMatch> calculateParameter(Frame& frame1, Frame& frame2);

	Mat rvec;
	Mat tvec;
	Mat inliers;
};



#endif
