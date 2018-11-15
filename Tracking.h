#ifndef TRACKING_H
#define TRACKING_H
#include "base.h"
#include "frame.h"

class Tracking
{

public:
    enum eTrackingState
    {
      SYSTEM_NOT_READY=-1,
      NO_IMAGE_YET=0,
      NOT_INITIALIZED=1,
      OK=2,
      LOST=3
    };
    
    eTrackingState mState;
    eTrackingState mLastState;
  
  
    Tracking();
    void grabImageRGBD(const cv::Mat& imRgb, const cv::Mat& imDepth, const double& timestamp);
    void grabImageMono(const cv::Mat& imRgb, const double& timestamp);
    void track();
    void checkKeyframe();
    
    
    //current frame
    Frame mCurrFrame;
    cv::Mat mImRGB;
    cv::Mat mImDepth;
    
};



#endif