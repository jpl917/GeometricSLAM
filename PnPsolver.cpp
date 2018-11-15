#include "PnPsolver.h"
#define GMS_MATCHER

//image(u,v,d) -> space(x,y,z)
Point3f point2dTo3d(Point3f& point,Camera& camera)
{
	Point3f res;
	res.z=(double)point.z/camera.scale;
	res.x=(point.x-camera.cx)*res.z/camera.fx;
	res.y=(point.y-camera.cy)*res.z/camera.fy;
	return res;
}


Point2d pixel2cam ( Point2f& p, Mat& K )
{
    return Point2d(
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 ));
}




vector<DMatch> PnPsolver::calculateParameter(Frame& frame1,Frame& frame2)
{
	Camera camera = frame1.camera;
	vector<DMatch> matches;
	BruteForceMatcher<HammingLUT> matcher;
	matcher.match(frame1.mDescriptors,frame2.mDescriptors,matches);
	cout<<"BF matches:" << matches.size()<<endl;
	
	vector<DMatch> goodMatches;
	
#ifdef GMS_MATCHER
	goodMatches = GmsMatch(frame1,frame2, matches);
#else
	goodMatches = FilterMatch(frame1,frame2,matches);
#endif
	
	if(goodMatches.size() == 0)return goodMatches;
	
	vector<Point3f> pts_obj;
	vector<Point2f> pts_img;

	for(size_t i=0;i<goodMatches.size();i++)
	{
		Point2f p=frame1.mvKeypoints[goodMatches[i].queryIdx].pt;
		float d=frame1.depth.ptr<float>(int(p.y))[int(p.x)];
		if(d==0)continue;
		pts_img.push_back(frame2.mvKeypoints[goodMatches[i].trainIdx].pt);
		Point3f pt(p.x,p.y,d);
		Point3f pd=point2dTo3d(pt,camera);
		pts_obj.push_back(pd);
	}

	Mat mK = Frame::K;
	Mat mDistCoeffs = Frame::distCoeffs;
	//cout<<mDistCoeffs<<endl;
	
	solvePnPRansac(pts_obj,pts_img,mK,Mat(),rvec,tvec,false,100,1.0,100,inliers);
	vector<DMatch> matchShow;
	for(size_t i=0;i<inliers.rows;i++)
	{		 
		matchShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
	}

	//cout<<"Find show matches:" << matchShow.size()<<endl;
	//cout<<"inliers: "<<inliers.rows<<endl;
	Mat R;
	Rodrigues(rvec,R);
	Mat t = (Mat_<double>(3,1)<<tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0));

	Mat img;
	img = frame2.rgb.clone();
	for(int i=0; i<goodMatches.size();i++)
	{
	  circle(img,frame1.mvKeypoints[goodMatches[i].queryIdx].pt,1,Scalar(255,0,0),2);
	  circle(img,frame2.mvKeypoints[goodMatches[i].trainIdx].pt,1,Scalar(0,0,255),2);
	}
	
	drawMatches(frame1.rgb,frame1.mvKeypoints,frame2.rgb,frame2.mvKeypoints,goodMatches,img);
	imshow("matchShow",img);
	cvWaitKey(1);
	
	
	//set Frame2 pose
	Mat Tcw = cv::Mat::eye(4,4,CV_32F);
	if(frame1.id==0)frame1.setPose(Tcw);
	R.copyTo(Tcw.rowRange(0,3).colRange(0,3));
	t.copyTo(Tcw.rowRange(0,3).col(3));
	frame2.setPose(Tcw);
	
	cout<<"Tcw:"<<Tcw<<endl;
	for(int i=0; i<goodMatches.size();i++)
	{
	  //cout<<frame1.feature_locations_3d_[goodMatches[i].queryIdx]<<" -- "
	   //   <<frame2.feature_locations_3d_[goodMatches[i].trainIdx]<<endl;
	}
	
	Mat t_x = (Mat_<double>(3,3) << 
	  0, 		-t.at<double>(2.0), t.at<double>(1,0),
	  t.at<double>(2,0),0,		    -t.at<double>(0,0),
	  -t.at<double>(1,0),t.at<double>(0,0), 0);
	
	//cout<<"t_R:"<<t_x*R<<endl;
	
	Mat  e = t_x*R;
	Mat  e1 = (Mat_<double>(3,3)<<e.at<double>(0,0),e.at<double>(0,1),e.at<double>(0,2),
	   e.at<double>(1,0),e.at<double>(1,1),e.at<double>(1,2),
	   e.at<double>(2,0),e.at<double>(2,1),e.at<double>(2,2)); 	 
	//cout<< mK.t().inv()*e1*mK.inv()<<endl;
	
	Mat K = Frame::K;
    
	for(DMatch m: goodMatches)
	{
	  Point2d pt1 = pixel2cam(frame1.mvKeypoints[m.queryIdx].pt,K);
	  Mat y1 = (Mat_<double>(3,1)<<pt1.x,pt1.y,1);
	  
	  Point2d pt2 = pixel2cam(frame2.mvKeypoints[m.trainIdx].pt,K);
	  Mat y2 = (Mat_<double>(3,1)<<pt2.x,pt2.y,1);
	  
	  Mat d = y2.t()*e1*y1;
	  if(d.at<double>(0,0)>0.01)   
	      cout<<"Epiploar Constraint:"<<d.at<double>(0,0)<<endl;
	}
	
	
	return matchShow;
}


