#ifndef POINT_CLOUD_MAPPING_H
#define POINT_CLOUD_MAPPING_H


#include "base.h"
#include <condition_variable>

class PointCloudMapping
{
public:
	PointCloudMapping();
	void viewer();
	
	
	std::shared_ptr<thread> viewerThread;
	
	PointCloud::Ptr globalMap;
	bool 		shutDownFlag=false;
	std::mutex 	shutDownMutex;
	
	condition_variable 	cv_keyframeUpdate;
	std::mutex  		keyframeUpdateMutex;
	
	std::mutex 			keyframeMutex;
	
	
	double resolution = 0.05;

};



#endif