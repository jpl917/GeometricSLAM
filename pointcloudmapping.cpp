#include "pointcloudmapping.h"

PointCloudMapping::PointCloudMapping()
{

	globalMap=boost::make_shared<PointCloud>();
	viewerThread=make_shared<thread>(bind(&PointCloudMapping::viewer,this));
	

}


void PointCloudMapping::viewer()
{
	pcl::visualization::CloudViewer viewer("viewer");
	while(1)
	{
		{
			unique_lock<mutex> lck_shutdown(shutDownMutex);
			if(shutDownFlag)break;
		}
		{
			unique_lock<mutex> lck_keyframeUpdate(keyframeUpdateMutex);
			cv_keyframeUpdate.wait(lck_keyframeUpdate);
		}
		
		{
			unique_lock<mutex> lck_keyframe(keyframeMutex);
			
		}
	}
	return;
}
