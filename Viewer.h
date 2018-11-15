#ifndef VIEWER_H
#define VIEWER_H

#include <pangolin/pangolin.h>

class Viewer
{
public:
	Viewer();
	void run();
	
	
	
private:
	float mImageWidth,mImageHeight;
	float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
};




#endif