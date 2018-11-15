#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include "base.h"
#include "SysParams.h"
#include "Camera.h"
using namespace std;

void loadImage(const string strAssociationFilename,  vector<double>& vdTimestamps,
	       vector<string>& vstrFilenamesRGB,  vector<string>& vstrFilenamesDepth)
{
  
  ifstream fin(strAssociationFilename);
  while(!fin.eof())
  {
      string s;
      getline(fin,s);
      if(!s.empty())
      {
	stringstream ss(s.c_str());
	double t;
	string sRgb, sDepth;
	ss>>t;
	vdTimestamps.push_back(t);
	ss>>sRgb>>t>>sDepth;
	vstrFilenamesRGB.push_back(sRgb);
	vstrFilenamesDepth.push_back(sDepth);
      }
  }
  return;
}



int main(int argc, char** argv)
{
  
    SysParams sysparams("/home/jpl/lines/f1.yaml");
    Camera camera(sysparams);
  
    string strAssociationFilename = "/home/jpl/lines/rgbd_dataset_freiburg1_xyz/associations.txt";
    int nImages = 0;
    vector<double> vdTimestamps;
    vector<string> vstrFilenamesRGB;
    vector<string> vstrFilenamesDepth;
    //load the filename of rgb and depth
    loadImage(strAssociationFilename,vdTimestamps, vstrFilenamesRGB, vstrFilenamesDepth);
    nImages = vstrFilenamesRGB.size();
    if(vstrFilenamesRGB.empty())return 0;
    if(vstrFilenamesRGB.size()!=vstrFilenamesDepth.size())return 0;
    
    //load image
    cv::Mat imRGB, imDepth;
    for(int i=0 ;i<nImages; i++)
    {
      imRGB = cv::imread(vstrFilenamesRGB[i]);
      imDepth = cv::imread(vstrFilenamesDepth[i]);
      double tstamp = vdTimestamps[i];
    }
    
    
    
    return 0;
}
