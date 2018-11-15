#ifndef CAMERA_H
#define CAMERA_H
#include "SysParams.h"


class Camera
{
public:
	float fx;
	float fy;
	float cx;
	float cy;
	float scale;
	
	float k1,k2,p1,p2,k3;
	
	Camera(){}
	
	Camera(const SysParams sysparams)
	{
	    fx = sysparams.fx;
	    fy = sysparams.fy;
	    cx = sysparams.cx;
	    cy = sysparams.cy;
	    scale = sysparams.scale;
	    
	    k1 = sysparams.k1;
	    k2 = sysparams.k2;
	    p1 = sysparams.p1;
	    p2 = sysparams.p2;	
	    k3 = sysparams.k3;	  
	}
};

#endif