#include "Tracking.h"


Tracking::Tracking(SysParams sysparam)
{
  
}


void Tracking::grabImageRGBD(const cv::Mat& imRgb, const cv::Mat& imDepth, const double& timestamp)
{
  mImRGB=imRgb;
  mImDepth=imDepth;
  mCurrFrame = Frame(timestamp,mImRGB,mImDepth);
}

void Tracking::grabImageMono(const cv::Mat& imRgb, const double& timestamp)
{
  mImRGB=imRgb;
  //mImDepth=imRgb;
  //mCurrFrame = Frame(timestamp,mImRGB);
}


void Tracking::track()
{
  
}


void Tracking::checkKeyframe()
{
  
}
    

