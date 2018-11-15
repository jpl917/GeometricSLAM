#ifndef SYSPARAMS_H
#define SYSPARAMS_H
#include "base.h"

class SysParams 
{
public:
  //camera
  double fx, fy, cx, cy, scale;
  double k1, k2, p1, p2, k3;
  
  //ORB
  float nFeatures;
  float scaleFactor;
  float nLevels;
  
  SysParams(){}
  
  SysParams(const string filename)
  {
      cv::FileStorage fparams(filename, cv::FileStorage::READ);
      if(!fparams.isOpened())return;
      
      fx = fparams["Camera.fx"];
      fy = fparams["Camera.fy"];
      cx = fparams["Camera.cx"];
      cy = fparams["Camera.cy"];
      scale = fparams["Camera.scale"];
      
      k1 = fparams["Camera.k1"];
      k2 = fparams["Camera.k2"];
      p1 = fparams["Camera.p1"];
      p2 = fparams["Camera.p2"];
      k3 = fparams["Camera.k3"];
      
      nFeatures = fparams["ORBextractor.nFeatures"];
      scaleFactor = fparams["ORBextractor.scaleFactor"];
      nLevels = fparams["ORBextractor.nLevels"];
      
      
      fparams.release();
  }
  
};




#endif