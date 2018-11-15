//g++ main.cpp -I /usr/include/python2.7 -L /usr/lib/python2.7/config-x86_64-linux-gnu -lpython2.7  
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <iostream>
#include <string>
#include <cmath>
#include <cv.h>
#include <highgui.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>  
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>

#include "utils.h"
#include "base.h"

using namespace std;
using namespace cv;


//tum3
double fx = 535.4;
double fy = 539.2;
double cx = 320.1;
double cy = 247.6;
double s  = 5000.0;

/*tum4
double fx = 525.0;
double fy = 525.0;
double cx = 319.5;
double cy = 239.5;
double s  = 5000.0;*/



PointT genPoint(Mat rgb, Mat depth, int i, int j, double _fx , double _fy, double _cx, double _cy, double _s)
{
	
	PointT point;
	double d=(double)depth.ptr<unsigned short>(i)[j]/_s;
	if(d<=1e-3||d>=10) return point;
	point.z =  d;
	point.x = (j - _cx) * d / _fx;
	point.y = (i - _cy) * d / _fy;
	point.b = rgb.ptr<uchar>(i)[j*3];
	point.g = rgb.ptr<uchar>(i)[j*3+1];
	point.r = rgb.ptr<uchar>(i)[j*3+2];
	
	
	//int u= (int)(point.y * _fy / point.z + _cy);
	//int v= (int)(point.x * _fx / point.z + _cx);
	
	//cout<<i<<" "<<j<<" "<<u<<" "<<v<<endl;
	//waitKey();
	return point;
}


void initPython(PyObject* &pNet, PyObject* &pFunc)
{
	Py_Initialize();//ini
	/*call python 0.5s*/
	string path = "/home/jpl/lines/pydensecrf/cmuFcnAlexnet";
	string chdir_cmd = string("sys.path.append(\"") + path + "\")";
	const char* cstr_cmd = chdir_cmd.c_str();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(cstr_cmd);
	
	PyObject* pModule = PyImport_ImportModule("inference1");
	if(!pModule)
	{
		cout<<"pModule Error"<<endl;
		return;
	}
	
	PyObject* pInit = PyObject_GetAttrString(pModule, "ini");
	if(!pInit)
	{
		cout<<"pIni Error"<<endl;
		return;
	}
	pNet = PyObject_CallObject(pInit, NULL);
	if(!pNet)
	{
		cout<<"pNet Error"<<endl;		
		return;
	}

	pFunc = PyObject_GetAttrString(pModule, "seg");
	if(!pFunc)
	{
		cout<<"pFunc Error"<<endl;		
		return;
	}
	
}


vector<double> seg(PyObject* pNet, PyObject* pFunc, string rgbname, string depthname, PointCloud::Ptr &cloud_all, PointCloud::Ptr &cloud)
{
	Camera tum1;
	tum1.fx = 517.306408;
	tum1.fy = 516.469215;
	tum1.cx = 318.643040;
	tum1.cy = 255.313989;
	tum1.scale = 5000.0;
	
	Camera tum2;
	tum2.fx = 520.908620;
	tum2.fy = 521.007327;
	tum2.cx = 325.141442;
	tum2.cy = 249.701764;
	tum2.scale = 5208.0;
	
	Camera tum3;
	tum3.fx = 535.4;
	tum3.fy = 539.2;
	tum3.cx = 320.1;
	tum3.cy = 247.6;
	tum3.scale = 5000.0;
	
	Frame frame1(0,rgbname,depthname,tum2);
	
	/*call segmentation ~50 ms GPU*/
	PyObject* arg0 = pNet;
	PyObject* arg1 = Py_BuildValue("s", rgbname.c_str());    
	PyObject* arg2 = Py_BuildValue("s", "tmp.png");
	PyObject* args = PyTuple_New(3); 
	PyTuple_SetItem(args, 0, arg0);    
	PyTuple_SetItem(args, 1, arg1);
	PyTuple_SetItem(args, 2, arg2);
	
	//MyTimer mytimer;
	//mytimer.start();
	PyObject* pRet = PyObject_CallObject(pFunc, args);
	//mytimer.end();

	
	/*genpoint ~53ms*/
	Mat rgb=imread(rgbname);
	Mat depth = imread(depthname,CV_LOAD_IMAGE_ANYDEPTH);
	//Mat depth;
	//depth_ori.convertTo(depth, CV_32F);	

	//PointCloud::Ptr cloud(new PointCloud);
	PointT point;

	if(pRet)
	{
		PyArrayObject* in_con = (PyArrayObject*)pRet;
		unsigned char* ptr = (unsigned char*)PyArray_DATA(in_con);
    	//int num_dim = PyArray_NDIM(in_con);
		npy_intp* pdim = PyArray_DIMS(in_con);
		//MyTimer mytimer;
		//mytimer.start();
		for(int i=0; i<pdim[0]*pdim[1]; i++)
		{
			point = genPoint(rgb, depth, i/pdim[1],i%pdim[1], fx,fy,cx,cy,s);
			if(point.z==0)continue;
			cloud_all->push_back(point);
			if((int)ptr[i]!=255)
			{
				cloud->push_back(point);
			}
			
		}
		//mytimer.end();
	}
	cout<<"Count:"<<cloud_all->points.size()<<endl;

	//ax+by+cz+d=0
	vector<double> c(4);
	if(cloud->points.size()==0)return c;

	/*calculate normal 8ms*/
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (1000);
	seg.setDistanceThreshold (0.01);
	seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);

	for(size_t i=0; i<4; i++) c[i]=coefficients->values[i];
	
	Mat rs=rgb.clone();
	PointT *pointT;
	for(size_t i=0; i<cloud_all->points.size(); i++)
	{
		pointT=&cloud_all->points[i];
		if(abs(c[0]*pointT->x + c[1]*pointT->y + c[2]*pointT->z + c[3])<=0.05)
		{
			pointT->r=255;	
			pointT->g=0;
			pointT->b=0;
			
			//int u= (int)(pointT->y * fy / pointT->z + cy);
			//int v= (int)(pointT->x * fx / pointT->z + cx);
			int u= (int)round(pointT->y * fy / pointT->z + cy);
			int v= (int)round(pointT->x * fx / pointT->z + cx);
			//if(u<0 || u>=480 ||v<0||v>=640)continue;
			//rs.ptr<uchar>(u)[v*3]=255;
		}
	}
	
	for(int k=0; k<frame1.lines.size();k++)
	{
		cv::Point2f left1 = frame1.lines[k].p;
		cv::Point2f left2 = frame1.lines[k].q;
		cv::line(rs,left1,left2,cv::Scalar(0,255,0),3);
	}
	
	
	cv::imshow("1",rs);
	cv::waitKey();
	cv:imwrite("1.png",rs);
	return c;

}

/*
int test_icp()
{
	string rgbname;
	string depthname;	
	string root = "/home/jpl/lines/build/data/";

	PyObject* pNet = NULL;
	PyObject* pFunc = NULL;
	initPython(pNet, pFunc);   //~1077ms

	PointCloud::Ptr cloud_all(new PointCloud);
	
	PointCloud::Ptr cloud1(new PointCloud);
	PointCloud::Ptr cloud2(new PointCloud);
	
	PointCloud::Ptr plane1(new PointCloud);
	PointCloud::Ptr plane2(new PointCloud);
	
	rgbname= root + "0rgb.png";
	depthname= root + "0depth.png";
	vector<double> c1 = seg(pNet, pFunc, rgbname, depthname, cloud1, plane1); 
	cout<<"Coefficients:"<<c1[0]<<" "<<c1[1]<<" "<<c1[2]<<" "<<c1[3]<<endl;
	
	rgbname= root + "1rgb.png";
	depthname= root + "1depth.png";
	vector<double> c2 = seg(pNet, pFunc, rgbname, depthname, cloud2, plane2); 
	cout<<"Coefficients:"<<c2[0]<<" "<<c2[1]<<" "<<c2[2]<<" "<<c2[3]<<endl;

	
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setInputCloud(plane1);
	icp.setInputTarget(plane2);
	PointCloud Final;
	icp.align(Final);
	cout << "has converged:" << icp.hasConverged() 
		<<" score: " <<icp.getFitnessScore() << endl;
	cout << icp.getFinalTransformation() <<endl;
  
	PointCloud::Ptr tmp(new PointCloud);
	pcl::transformPointCloud( *cloud1, *tmp, icp.getFinalTransformation()); //pose = Twc
	*cloud_all += *tmp;
	*cloud_all += *cloud2;
	//visualization
	if(cloud_all->points.size()!=0)
	{
		pcl::visualization::CloudViewer viewer("viewer");
		viewer.showCloud(cloud_all);
		while(!viewer.wasStopped()){}
	}
	Py_Finalize();
	return 0;
}*/


int main(int argc, char** argv)
{
	
	string rgbname;
	string depthname;	
	string root = "/home/jpl/lines/build/data/";
	if(argc!=3){
		rgbname= root + "1rgb.png";
		depthname= root + "1depth.png";
	}
	else{
		rgbname=root + argv[1];
		depthname=root + argv[2];
	}

	PyObject* pNet = NULL;
	PyObject* pFunc = NULL;
	initPython(pNet, pFunc);   //~1077ms

	PointCloud::Ptr cloud_all(new PointCloud);
	PointCloud::Ptr plane(new PointCloud);
	vector<double> c = seg(pNet, pFunc, rgbname, depthname, cloud_all, plane); 
	cout<<"Coefficients:"<<c[0]<<" "<<c[1]<<" "<<c[2]<<" "<<c[3]<<endl;

	//visualization
	if(cloud_all->points.size()!=0)
	{
		pcl::visualization::CloudViewer viewer("viewer");
		viewer.showCloud(cloud_all);
		while(!viewer.wasStopped()){}
	}
	Py_Finalize();
	return 0;
}
