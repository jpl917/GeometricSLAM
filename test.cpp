#include <pangolin/pangolin.h>
#include "utils.h"
#include "base.h"
#include "PnPsolver.h"
#include "gms_matcher.h"
#include <cmath>
#include <map>
#include <iomanip>
#include <opencv2/core/eigen.hpp>

using namespace std;

Eigen::Isometry3d cvMat2Eigen(Mat& rvec,Mat& tvec)
{
      Mat R;
      cv::Rodrigues(rvec,R);
      Eigen::Matrix3d r;
      cv::cv2eigen(R,r);

      Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
      Eigen::AngleAxisd angle(r);
      T=angle;

      T(0,3)=tvec.at<float>(0,0);
      T(1,3)=tvec.at<float>(1,0);
      T(2,3)=tvec.at<float>(2,0); 
       
      //Eigen::Quaterniond q(r);
      //cout<<"Quaterniond"<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;
      
      return T;
}

Eigen::Matrix4f cvMat2EigenMatrix4f(Mat& rvec,Mat& tvec)
{
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      
      
      Mat R;
      Rodrigues(rvec,R);
      Mat t = (Mat_<double>(3,1)<<tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0));
     
      Mat Tcw = cv::Mat::eye(4,4,CV_32F);
      R.copyTo(Tcw.rowRange(0,3).colRange(0,3));
      t.copyTo(Tcw.rowRange(0,3).col(3));
      cv::cv2eigen(Tcw,T);

      return T;
}

PointCloud::Ptr joinPointCloud(PointCloud::Ptr cloud,Frame frame,Eigen::Isometry3d& T)
{
      PointCloud::Ptr newcloud=frame.img2cloud();
      
      //cout<<"Join PointCloud."<<endl;
      PointCloud::Ptr output(new PointCloud);
      pcl::transformPointCloud(*cloud,*output,T.matrix());
      *newcloud+=*output;
      return newcloud;
      //pcl::io::savePCDFile("1.pcd",*output);

}

cv::Mat drawInlier(Frame &f1, Frame &f2, vector<vector<int>>& matches)
{
    cv::Mat src1,src2;
    f1.rgb.copyTo(src1);
    f2.rgb.copyTo(src2);
    const int height = max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    cv::Mat output(height, width, CV_8UC3,cv::Scalar(0,0,0));
    src1.copyTo(output(cv::Rect(0,0,src1.cols, src1.rows)));
    src2.copyTo(output(cv::Rect(src1.cols,0,src2.cols, src2.rows)));

    return output;
}


int testline()
{
    Camera tum1;
    tum1.fx = 517.306408;
    tum1.fy = 516.469215;
    tum1.cx = 318.643040;
    tum1.cy = 255.313989;
    tum1.scale = 5000.0;
    
    Camera camera1;
    camera1.fx = 525;
    camera1.fy = 525;
    camera1.cx = 319.5;
    camera1.cy = 239.5;
    camera1.scale = 5000.0;
    
    Camera tum3;
    tum3.fx = 535.4;
    tum3.fy = 539.2;
    tum3.cx = 320.1;
    tum3.cy = 247.6;
    tum3.scale = 5000.0;
    
    SystemParameters sysPara;

    //Frame f1(0.0, "../data/1.png","../data/d1.png",tum3);
    //Frame f2(0.0, "../data/3.png","../data/d3.png",tum3);
    
    Frame f1(0.0, "../data/img1.png","../data/depth1.png",tum1);
    Frame f2(0.0, "../data/img3.png","../data/depth3.png",tum1);

    //Frame f1(0.0, "../pairs/pair1-scn2/normal_rgb.png","../pairs/pair1-scn2/normal_depth.png",camera1);
    //Frame f2(0.0, "../pairs/pair1-scn2/dark_1_rgb.png","../pairs/pair1-scn2/dark_1_depth.png",camera1);
    
    
    vector<DMatch> bf_matches;
    BruteForceMatcher<HammingLUT> matcher;
    matcher.match(f1.mDescriptors,f2.mDescriptors,bf_matches);
    vector<DMatch> pt_matches = GmsMatch(f1,f2, bf_matches);
    //vector<DMatch> pt_matches = FilterMatch(f1,f2, bf_matches);
    
    Mat img;
    img = f2.rgb.clone();
    for(int i=0; i<pt_matches.size();i++)
    {
		circle(img,f1.mvKeypoints[pt_matches[i].queryIdx].pt,1,Scalar(255,0,0),2);
		circle(img,f2.mvKeypoints[pt_matches[i].trainIdx].pt,1,Scalar(0,0,255),2);
    }
    drawMatches(f1.rgb,f1.mvKeypoints,f2.rgb,f2.mvKeypoints,pt_matches,img);
    imshow("matchShow",img);
	imwrite("1111.png",img);
    waitKey(0);
    


    vector<vector<int>> matches;
    matchLine(f1.lines,f2.lines,matches);
    vector<DMatch> ln_matches;
    for(int i=0; i<matches.size(); i++)
    {
		int p1 = matches[i][0];
		int p2 = matches[i][1];
		if(!f1.lines[p1].haveDepth || !f2.lines[p2].haveDepth)continue;
		double len1 = cv::norm(f1.lines[p1].line3d.A - f1.lines[p1].line3d.B);
		double len2 = cv::norm(f2.lines[p2].line3d.A - f2.lines[p2].line3d.B);
		//if(abs(len1-len2)>0.03)continue;
		DMatch tmp(matches[i][0],matches[i][1],0);
		ln_matches.push_back(tmp);
    }
   
   cv::Mat   t1,t2;
   t1=f1.rgb.clone();
   t2=f2.rgb.clone();
   cout<<"f1 Lines:"<<f1.lines.size()<<"  f2 Lines: "<<f2.lines.size()<<"  Original Match: "<<matches.size()<<"  Line Match: "<<ln_matches.size()<<endl;
   for(int i=0; i<f1.lines.size();i++)
   {
       cv::Point2f left1 = f1.lines[i].p;
       cv::Point2f left2 = f1.lines[i].q;
       cv::line(t1,left1,left2,cv::Scalar(0,0,255),3);
   }
   cvNamedWindow("output1");
   cv::imshow("output1", t1);
   
   for(int i=0; i<f2.lines.size();i++)
   {
       cv::Point2f right1 = f2.lines[i].p ;
       cv::Point2f right2 = f2.lines[i].q ;
       cv::line(t2,right1,right2,cv::Scalar(0,0,255),3);
   }
   cvNamedWindow("output2");
   cv::imshow("output2", t2);
    
    cv::Mat output = drawInlier(f1,f2,matches);
    cv::Mat src1,src2;
    f1.rgb.copyTo(src1);
    f2.rgb.copyTo(src2);
    srand((unsigned int)time(NULL));
    for(int i=0; i<ln_matches.size();i++)
    {
		int p = ln_matches[i].queryIdx;
		int q = ln_matches[i].trainIdx;
		cv::Point2f left1 = f1.lines[p].p;
		cv::Point2f left2 = f1.lines[p].q;
		cv::Point2f right1 = f2.lines[q].p + cv::Point2d((double)src1.cols,0.f);
		cv::Point2f right2 = f2.lines[q].q + cv::Point2d((double)src1.cols,0.f);
		cv::line(output,left1,left2,cv::Scalar(0,0,255),2);
		cv::line(output,right1,right2,cv::Scalar(0,0,255),2);
		int r = rand()%255;
		int g = rand()%255;
		int b = rand()%255;
		cv::line(output,left1,right1,cv::Scalar(b,g,r),2);
 	
    }
    cv::imshow("output", output);
    cvWaitKey();
    
    //PnPsolver pnpsolver;
    //pnpsolver.calculateParameter(f1,f2);   
    
    //double norm = fabs(min(cv::norm(pnpsolver.rvec), 2*M_PI-cv::norm(pnpsolver.rvec)))+ fabs(cv::norm(pnpsolver.tvec));
    //cout<<"norm:"<<norm<<endl;
    
    
    Eigen::Matrix4f T ;//= cvMat2EigenMatrix4f(pnpsolver.rvec,pnpsolver.tvec);
    //cvMat2Eigen(pnpsolver.rvec,pnpsolver.tvec);
    
    vector<DMatch> tmp;
    pt_matches = tmp;
    getTransformFromHybridMatchesG2O(&f1,&f2,pt_matches,ln_matches,T,10,sysPara);
    //getTransformFromMatchesG2O(&f1,&f2,pt_matches,T,10);
    
    //bool valid;
    //T = getTransform_Lns_Pts_pcl(&f1,&f2,pt_matches,ln_matches,valid);
    
    //vector<DMatch> out_pt_matches;
    //vector<DMatch> out_ln_matches;
    //float inlier_rmse;
    //getTransform_PtsLines_ransac(&f1,&f2,pt_matches,ln_matches,
    //			 out_pt_matches,out_ln_matches,T, inlier_rmse, sysPara);
	

    cout<<"Result:"<<T<<endl;

    return 0;
}



Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d(
      ( p.x - K.at<float> ( 0,2 ) ) / K.at<float> ( 0,0 ),
      ( p.y - K.at<float> ( 1,2 ) ) / K.at<float> ( 1,1 ));
}


int testPnPsolver()
{    
    Camera camera;
    camera.fx = 517.306408;
    camera.fy = 516.469215;
    camera.cx = 318.643040;
    camera.cy = 255.313989;
    camera.scale = 5000.0;
    
    camera.k1=0.262383;
    camera.k2=-0.953104;
    camera.p1=-0.005358;
    camera.p2= 0.002628;
    camera.k3= 1.163314;


    Frame f1(0.0,"../data/img1.png","../data/depth1.png",camera);
    Frame f2(0.0,"../data/img2.png","../data/depth2.png",camera);
    
    PointCloud::Ptr cloud11=f1.img2cloud(); 
    pcl::visualization::CloudViewer viewer1("viewer");
    viewer1.showCloud(cloud11);
    while(!viewer1.wasStopped()){}
    return 0;
    
    PnPsolver pnpsolver;
    vector<DMatch> matches = pnpsolver.calculateParameter(f1,f2);   
    
    Eigen::Isometry3d T = cvMat2Eigen(pnpsolver.rvec,pnpsolver.tvec);
    cout<<"Rotation:"<<T.rotation()<<endl<<"Translation:"<<""<<T.translation()<<endl;
    //cout<<pnpsolver.rvec<<endl;

    Mat R;
    cv::Rodrigues(pnpsolver.rvec,R);
    Mat t = (Mat_<double>(3,1)<<pnpsolver.tvec.at<double>(0,0),pnpsolver.tvec.at<double>(1,0),pnpsolver.tvec.at<double>(2,0));
    cout<<"t:"<<t<<endl;
	
	
    Mat t_x = (Mat_<double>(3,3) << 
      0, 		-t.at<double>(2.0), t.at<double>(1,0),
      t.at<double>(2,0),0,		    -t.at<double>(0,0),
      -t.at<double>(1,0),t.at<double>(0,0), 0);
    
   // cout<<"t_R:"<<t_x*R<<endl;
    
    Mat K = f1.K;
    for(DMatch m: matches)
    {
      Point2d pt1 = pixel2cam(f1.mvKeypoints[m.queryIdx].pt,K);
      Mat y1 = (Mat_<double>(3,1)<<pt1.x,pt1.y,1);
      //cout<<y1<<endl;
      
      Point2d pt2 = pixel2cam(f2.mvKeypoints[m.trainIdx].pt,K);
      Mat y2 = (Mat_<double>(3,1)<<pt2.x,pt2.y,1);
      
      Mat d = y2.t()*t_x*R*y1;
      if(d.at<double>(0,0)>0.01)      
	cout<<"Epiploar Constraint:"<<d<<endl;
      
    }
    
    PointCloud::Ptr cloud1 ( new PointCloud() ); 
    PointCloud::Ptr output(new PointCloud());
    PointCloud::Ptr tmp(new PointCloud());
    pcl::VoxelGrid<PointT> voxel;
    double gridsize = 0.05; 
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    
    cloud1 = f1.img2cloud();
    output = joinPointCloud(cloud1,f2,T);

    pcl::visualization::CloudViewer viewer("viewer");
    viewer.showCloud(output);
    while(!viewer.wasStopped()){}
    
    return 0;
}

vector<DMatch> doMatch(Frame frame1, Frame frame2)
{
    vector<DMatch> matches;
    BruteForceMatcher<HammingLUT> matcher;
    matcher.match(frame1.mDescriptors,frame2.mDescriptors,matches);
    
    cout<<"Original matches:" << matches.size()<<endl;

    double minDist = 1e5, maxDist = 0;
    for(size_t i=0;i<matches.size();i++)
    {
	  double dist = matches[i].distance;
	  minDist = dist < minDist ? dist: minDist;
	  maxDist = dist > maxDist ? dist: maxDist;
    }
    cout<<"MinDist:"<<minDist<<endl;
    cout<<"MaxDist:"<<maxDist<<endl;
    
    vector<DMatch> goodMatches;
    for(size_t i=0;i<matches.size();i++)
    {
	    //if(matches[i].distance<0.6*maxDist&&matches[i].distance < 15*minDist)
	    if ( matches[i].distance <= max ( 2*minDist, 20.0 ) )
	    {
		    goodMatches.push_back(matches[i]);
	    }
    }   
    cout<<"Good Matches:"<<goodMatches.size()<<endl;
    return goodMatches;
}

//Each time pick up eight points
vector<vector<int>> generateRandomSets(int mMaxIterations, int N)
{
  vector<int> vAvaliableIndices(N);
  for(int i=0; i<N; i++)
  {
    vAvaliableIndices[i]=i;
  }
  vector<vector<int>> mvSets = vector<vector<int>>(mMaxIterations, vector<int>(8,0));
  srand(time(NULL));
  
  for(int i=0; i< mMaxIterations; i++)
  {
    for(int j=0; j<8; j++)
    {
      int randi = random()%(vAvaliableIndices.size());
      int idx = vAvaliableIndices[randi];
      mvSets[i][j] = idx;
      vAvaliableIndices[randi]=vAvaliableIndices.back();
      vAvaliableIndices.pop_back();
    }
  }
  return mvSets;
}


cv::Mat computeF21(const vector<Point2f> &vP1, const vector<Point2f> &vP2)
{
    vector<int> vAvaliableIndices(vP1.size());
    for(int i=0; i<vP1.size(); i++)
    {
      vAvaliableIndices[i]=i;
    }
    vector<int> mvSets;
    mvSets.resize(8);
    srand(time(NULL));

    for(int i=0; i<8; i++)
    {
      int randi = random()%(vP1.size());
      int idx = vAvaliableIndices[randi];
      mvSets[i] = idx;
      vAvaliableIndices[randi]=vAvaliableIndices.back();
      vAvaliableIndices.pop_back();
    }
  
    int N = vP1.size();
    cv::Mat A(N,9,CV_32F);
    for(int i=0; i<8; i++)
    {
	int t = mvSets[i];
	const float u1 = vP1[t].x;
	const float v1 = vP1[t].y;
	const float u2 = vP2[t].x;
	const float v2 = vP2[t].y;
	
	A.at<float>(i,0) = u2*u1;
	A.at<float>(i,1) = u2*v1;
	A.at<float>(i,2) = u2;
	A.at<float>(i,3) = v2*u1;
	A.at<float>(i,4) = v2*v1;
	A.at<float>(i,5) = v2;
	A.at<float>(i,6) = u1;
	A.at<float>(i,7) = v1;
	A.at<float>(i,8) = 1;
    }
    
    cv::Mat u,w,vt;
    //cout<<"A:"<<A<<endl;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    //cout<<"u:"<<u<<endl;
    //cout<<"w:"<<w<<endl;
    //cout<<"vt:"<<vt<<endl;
    cv::Mat Fpre = vt.row(8).reshape(0,3);
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);  
    //w.at<float>(2)=0;
    cv::Mat F21 = u*cv::Mat::diag(w)*vt;
    return F21;
}


void findFundamental(float &score, Mat& F21)
{
    return;
}

void pose_estimate_2d2d()
{
    Camera camera;
    camera.fx = 517.306408;
    camera.fy = 516.469215;
    camera.cx = 318.643040;
    camera.cy = 255.313989;
    camera.scale = 5000.0;
    
    camera.k1=0.262383;
    camera.k2=-0.953104;
    camera.p1=-0.005358;
    camera.p2= 0.002628;
    camera.k3= 1.163314;


    Frame f1(0.0,"../data/img1.png","../data/depth1.png",camera);
    Frame f2(0.0,"../data/img2.png","../data/depth2.png",camera);
    
    BruteForceMatcher<HammingLUT> matcher;
    //FlannBasedMatcher  matcher;
    vector<DMatch> matches;
    matcher.match(f1.mDescriptors,f2.mDescriptors,matches);
    
    vector<DMatch> goodMatches = doMatch(f1,f2);
     
    vector<Point2f> points1;
    vector<Point2f> points2;
    vector<vector<int>> vmatches;
    //cv::Mat output = drawInlier(f1,f2,vmatches);
    
    Mat res;
    drawMatches(f1.rgb,f1.mvKeypoints,f2.rgb,f2.mvKeypoints,goodMatches,res,Scalar::all(-1),Scalar::all(-1),
      vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("result", res);
    cvWaitKey(); 
    
    for(int i=0; i< goodMatches.size(); i++)
    {
	 points1.push_back(f1.mvKeypoints[goodMatches[i].queryIdx].pt);
	 points2.push_back(f2.mvKeypoints[goodMatches[i].trainIdx].pt);
    }
    
    Mat F = findFundamentalMat(points1,points2,CV_FM_8POINT);
    cout<<"Fundamental: "<<F<<endl;
    
    //cv::Mat  F1 = computeF21(points1,points2);
    //cout<<"ComputeF21:"<<F1<<endl;
    
    //cv::Mat H = findHomography(points1, points2,RANSAC,3,noArray());
    //cout<<"H:"<<H<<endl;
}

void test()
{
	SysParams sysparams("/home/jpl/lines/TUM2.yaml");
 	SystemParameters sysPara;   //line parameters
 	Camera camera(sysparams);
 	
 	Frame f1(0.0,"../data/img1.png","../data/depth1.png",camera);
 	cout<<Frame::mnMinX<<" "<<Frame::mnMaxX<<" "<<Frame::mnMinY<<" "<<Frame::mnMaxY<<endl;
	return;
}

void testDBoW()
{
	
	SysParams sysparams("/home/jpl/lines/TUM2.yaml");
 	SystemParameters sysPara;   //line parameters
 	Camera camera(sysparams);
 	
 	Frame f1(0.0,"../data/img1.png","../data/depth1.png",camera);
	
	ORBVocabulary *mpORBVocabulary= new ORBVocabulary();
    mpORBVocabulary->loadFromTextFile("../Vocabulary/ORBvoc.txt");
	cout<<*mpORBVocabulary<<endl;
	
	
	DBoW2::BowVector  			mBowVec;
    DBoW2::FeatureVector		mFeatureVec;
	vector<cv::Mat> 			vDescTmp;
	vDescTmp.reserve(f1.mDescriptors.rows);
	for(int i=0; i<f1.mDescriptors.rows;i++)
	{
		vDescTmp.push_back(f1.mDescriptors.row(i));
	}
	
	mpORBVocabulary->transform(vDescTmp,mBowVec,mFeatureVec,4);
	cout<<mBowVec.size()<<endl;
	cout<<mFeatureVec.size()<<endl;
	cout<<mBowVec.begin()->first<<endl;
	
	for(int i=140 ;i<165; i++)
		cout<<i<<" "<<mBowVec[i]<<endl;
	return;
}

void test1()
{
	cv::Mat rgb = cv::imread("sample.png",-1);
	cv::Mat rgb1 = cv::imread("sample-1.png",-1);
	cv::Mat gray;
	cv::cvtColor(rgb,gray,CV_BGR2GRAY);
	
	int n;
	LS* ls = callEDLines(gray, &n);
	for(int i=0; i<n; i++) 
	{
		// store output to lineSegments 
		if ((ls[i].sx-ls[i].ex)*(ls[i].sx-ls[i].ex) +(ls[i].sy-ls[i].ey)*(ls[i].sy-ls[i].ey) 
			> 900) {
			cv::Point2f p1 = cv::Point2d(ls[i].sx,ls[i].sy);
			cv::Point2f p2 = cv::Point2d(ls[i].ex,ls[i].ey);
			cv::line(rgb1,p1,p2,cv::Scalar(0,255,0),2);
		}
	}
	imshow("1",rgb1);
	imwrite("sample-2.png",rgb1);
	waitKey();
	
	return;
}


void getEquation(PointCloud::Ptr cloud, vector<double>& c)
{
	//cout<<cloud->size()<<endl;
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
	//cout<<cloud->size()<<endl;
	c.reserve(4);
	for(size_t i=0; i<4; i++) 
	{	
		c[i]=coefficients->values[i];
		cout<<c[i]<<" ";
	}
	cout<<endl;
	return;
}


void test2(string rgbname, string depthname, int id)
{
	Camera tum[4];
	tum[0].fx = 517.306408;
	tum[0].fy = 516.469215;
	tum[0].cx = 318.643040;
	tum[0].cy = 255.313989;
	tum[0].scale = 5000.0;
	
	tum[1].fx = 520.908620;
	tum[1].fy = 521.007327;
	tum[1].cx = 325.141442;
	tum[1].cy = 249.701764;
	tum[1].scale = 5208.0;

    tum[2].fx = 535.4;
    tum[2].fy = 539.2;
    tum[2].cx = 320.1;
    tum[2].cy = 247.6;
    tum[2].scale = 5000.0;
	
    tum[3].fx = 525;
    tum[3].fy = 525;
	tum[3].cx = 319.5;
	tum[3].cy = 239.5;
	tum[3].scale = 5000.0;
	
    SystemParameters sysPara;
	
    //Frame frame(0.0, "data/1rgb.png","data/1depth.png",tum[id]);
	Frame frame(0.0, "data/"+rgbname,"data/"+depthname,tum[id-1]);
	
	Camera myTum = frame.camera;

	PointCloud::Ptr cloud=frame.img2cloud();	
	cout<<"size:"<<cloud->size()<<endl;
	
// 	pcl::VoxelGrid<PointT> voxel;
// 	double gridsize = 0.03; 
// 	voxel.setLeafSize( gridsize, gridsize, gridsize );
// 	PointCloud::Ptr tmp(new PointCloud());
// 	voxel.setInputCloud( cloud );
//     voxel.filter( *tmp );
//     cloud.swap(tmp);
// 	cout<<"size:"<<cloud->size()<<endl;
	
// 	pcl::visualization::CloudViewer viewer("viewer");
//     viewer.showCloud(cloud);
// 	pcl::io::savePCDFile("tmp.pcd",*cloud);
//     while(!viewer.wasStopped()){}
	
	
	//segmentation
	MyTimer mytimer;
	mytimer.start();
	pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree = 
		boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA> > (new pcl::search::KdTree<pcl::PointXYZRGBA>);
	pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod (tree);
	normal_estimator.setInputCloud (cloud);
	normal_estimator.setKSearch (100);
	//normal_estimator.setRadiusSearch (0.05);
	normal_estimator.compute (*normals);
	{
			pcl::visualization::PCLVisualizer viewer("PCL Viewer");
			//viewer.setBackgroundColor (0.0, 0.0, 0.5);
			viewer.addPointCloudNormals<pcl::PointXYZRGBA,pcl::Normal>(cloud, normals);
			while (!viewer.wasStopped ())
			{
				viewer.spinOnce ();
			}
	}

	pcl::IndicesPtr indices (new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZRGBA> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0, 10.0);
	pass.filter (*indices);
	//cout<<"size2:"<<cloud->size()<<endl;;

	pcl::RegionGrowing<pcl::PointXYZRGBA, pcl::Normal> reg;
	reg.setMinClusterSize (1000);
	reg.setMaxClusterSize (1000000);
	reg.setSearchMethod (tree);
	reg.setNumberOfNeighbours (30);
	reg.setInputCloud (cloud);
	reg.setIndices (indices);
	reg.setInputNormals (normals);
	reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
	reg.setCurvatureThreshold (0.10);

	std::vector <pcl::PointIndices> clusters;
	reg.extract(clusters);
	mytimer.end();

	std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
	vector<double> c;
	srand(unsigned(time(NULL)));
	Mat rs= frame.rgb.clone();
	for(int i=0; i<clusters.size(); i++)
	{
		//std::cout << "Cluster " <<i<<" "<<clusters[i].indices.size () << " points." << endl;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr each_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
		//each_plane->resize(clusters[i].indices.size ());
		int r = rand()%255;
		int g = rand()%255;
		int b = 200;
		for (size_t j = 0; j < clusters[i].indices.size (); ++j)
		{

			pcl::PointXYZRGBA point;
			point.x = cloud->points[clusters[i].indices[j]].x;
	  		point.y = cloud->points[clusters[i].indices[j]].y;
	 		point.z = cloud->points[clusters[i].indices[j]].z;
			point.r = r;
			point.g = g;
			point.b = b;
			point.a = 0.0;
			each_plane->push_back(point);
			cloud_plane->push_back(point);
			
			int u= (int)(point.y * myTum.fy / point.z + myTum.cy);
			int v= (int)(point.x * myTum.fx / point.z + myTum.cx);
				
			rs.ptr<uchar>(u)[v*3]=r;
			rs.ptr<uchar>(u)[v*3+1]=g;
			rs.ptr<uchar>(u)[v*3+2]=b;
			
			//cloud->points[clusters[i].indices[j]].r = r;
			//cloud->points[clusters[i].indices[j]].g = g;
			//cloud->points[clusters[i].indices[j]].b = b;
			//cloud->points[clusters[i].indices[j]].a = 0.0;
			//cloud->erase(index+inliers->indices[i]);
		}
		getEquation(each_plane,c);

// 		for(size_t j=0; j<cloud->points.size(); j++)
// 		{
// 			double x = cloud->points[j].x;
// 			double y = cloud->points[j].y;
// 			double z = cloud->points[j].z;
// 			//cout<<x<<" "<<y<<" "<<z<<endl;
// 			if(abs(c[0]*x + c[1]*y + c[2]*z + c[3])<=0.05)
// 			{
// 				pcl::PointXYZRGBA p;
// 				p.x = x;
// 				p.y = y;
// 				p.z = z;
// 				p.r = r;
// 				p.g = g;
// 				p.b = b;
// 				p.a = 0.0;
// 				each_plane->push_back(p);
// 				cloud_plane->push_back(p);
// 				int u= (int)(pointT->y * fy / pointT->z + cy);
// 				int v= (int)(pointT->x * fx / pointT->z + cx);
// 				
// 				rs.ptr<uchar>(u)[v*3]=255;
// 			}
// 		}
		//pcl::visualization::CloudViewer viewer("viewer");
		//viewer.showCloud(each_plane);
		//while(!viewer.wasStopped()){}
	}
	
	
	//pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
	pcl::visualization::CloudViewer viewer ("Cluster viewer");
	viewer.showCloud(cloud_plane);
	viewer.runOnVisualizationThread(setPCLBackGround);
	while (!viewer.wasStopped ()){}
	
	//pcl::visualization::PCLVisualizer viewer("viewer");
	//viewer.setBackgroundColor(1.0,1.0,1.0);
	//viewer.showCloud(cloud_plane,"viewer");
	
	
	//cv::imshow("res",rs);
	//cv::waitKey();
	//cv:imwrite("plane.png",rs);
	
	/**/
	return;
}

Eigen::Matrix4d toTransformation(double tx,double ty, double tz,double qx,double qy, double qz, double qw)
{
	// 1 - 2*qy2 - 2*qz2	2*qx*qy - 2*qz*qw	2*qx*qz + 2*qy*qw
	// 2*qx*qy + 2*qz*qw	1 - 2*qx2 - 2*qz2	2*qy*qz - 2*qx*qw
	// 2*qx*qz - 2*qy*qw	2*qy*qz + 2*qx*qw	1 - 2*qx2 - 2*qy2
	Eigen::Matrix4d trans;
	trans<< 	1-2*qy*qy-2*qz*qz,	2*qx*qy - 2*qz*qw,	2*qx*qz + 2*qy*qw, tx,
			2*qx*qy + 2*qz*qw,	1-2*qx*qx-2*qz*qz,	2*qy*qz - 2*qx*qw, ty,
			2*qx*qz - 2*qy*qw,	2*qy*qz + 2*qx*qw,	1-2*qx*qx-2*qy*qy, tz,
			0,                  0,                  0,                 1.0;
	return trans;
}

PointCloud::Ptr img2cloud(Mat rgb, Mat depth, Camera camera)
{
    PointCloud::Ptr cloud(new PointCloud);
    for(int i=0;i<depth.rows;i+=3)
    {
		for(int j=0;j<depth.cols;j+=3)
		{
			double d=(double)depth.ptr<unsigned short>(i)[j]/camera.scale;
			//cout<<d<<endl;
			if(d <= 1e-2||d>=10)continue;
			PointT p;
			p.z= d;
			p.x=(j-camera.cx)*p.z/camera.fx;
			p.y=(i-camera.cy)*p.z/camera.fy;
			
			p.b=rgb.ptr<uchar>(i)[j*3];
			p.g=rgb.ptr<uchar>(i)[j*3+1];
			p.r=rgb.ptr<uchar>(i)[j*3+2];
			

			cloud->points.push_back(p);
		}
    }
   
    cloud->width=cloud->points.size();
    cloud->height=1;
    cloud->is_dense=false;
    
    return cloud;
}
 
int reconstruction(string dir)
{
	
	pcl::visualization::CloudViewer viewer("viewer");
	cv::namedWindow("rgb");
	cv::namedWindow("depth");
	cv::Mat img1=imread("0.png",-1);
	cv::imshow("rgb",img1);
    cv::imshow("depth",img1);
	Camera tum3;
    tum3.fx = 535.4;
    tum3.fy = 539.2;
    tum3.cx = 320.1;
    tum3.cy = 247.6;
    tum3.scale = 5000.0;
	
	PointCloud::Ptr globalMap ( new PointCloud() ); 
	PointCloud::Ptr tmp(new PointCloud());
	
	waitKey();
	
	vector<Eigen::Matrix4d> poses;
	//ifstream fin(dir+"/res/CameraTrajectory.txt");
	ifstream fin(dir+"jpl.txt");
	pcl::VoxelGrid<PointT> voxel;
	double gridsize = 0.02; 
	voxel.setLeafSize( gridsize, gridsize, gridsize );
	while(!fin.eof())
	{
		string s;
		getline(fin,s);
		//if(s[0]=='#')continue;
		if(!s.empty())
		{
			stringstream ss(s.c_str());
			double t,tx,ty,tz,qx,qy,qz,qw;
			double t1,t2;
			string s1,s2;
			fin>>t>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
			fin>>t1>>s1>>t2>>s2;
			Eigen::Matrix4d pose=toTransformation(tx,ty,tz,qx,qy,qz,qw);
			poses.push_back(pose);
			Mat rgb=cv::imread(dir+s1, CV_LOAD_IMAGE_UNCHANGED);
			Mat depth=cv::imread(dir+s2,CV_LOAD_IMAGE_ANYDEPTH);
			
			Mat rs=rgb.clone();
			Frame frame(0,rgb,depth,tum3);
			for(int k=0; k<frame.lines.size();k++)
			{
				cv::Point2f left1 = frame.lines[k].p;
				cv::Point2f left2 = frame.lines[k].q;
				cv::line(rs,left1,left2,cv::Scalar(0,255,0),3);
			}
			for(int k=0; k<frame.mvKeypoints.size();k++)
			{
				circle(rs,frame.mvKeypoints[k].pt,1,Scalar(0,0,255),2);
			}
			//frame.planeExtraction();
			cv::imshow("rgb",rs);
			cv::imshow("depth",depth);
			cv::waitKey(20);
			
			PointCloud::Ptr cloud = img2cloud(rgb,depth,tum3);
		
			pcl::transformPointCloud( *cloud, *tmp, pose); //pose = Twc
			*globalMap += *tmp;
			voxel.setInputCloud( globalMap );
			voxel.filter( *tmp );
			globalMap.swap(tmp);
			viewer.showCloud(globalMap);
			cloud->clear();
			tmp->clear();
		}
	}
	pcl::io::savePCDFile("global.pcd",*globalMap);
	

    
	//viewer.runOnVisualizationThread(setPCLBackGround);
    while(!viewer.wasStopped()){}
	
	fin.close();
	
	return 0;
}


int reconstruction_pangolin(string dir)
{
	
	float mImageWidth = 640;
	float mImageHeight = 480;
	float mViewpointX = 5;
	float mViewpointY = 5;
	float mViewpointZ = 5;
	float mViewpointF = 500;
	
	pangolin::CreateWindowAndBind("AR demo",640,480);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	
	pangolin::OpenGlRenderState s_cam(
			pangolin::ProjectionMatrix(1024,768,535.4,539.2,320.1,247.6,0.01,1000),
			pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,1.0, 0.0)
			);
			
	pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(0), 1.0, -640.0f/480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));			
			//.SetLock(pangolin::LockLeft, pangolin::LockTop);
			
	pangolin::View& d_image = pangolin::Display("image")
		.SetBounds(0,1.0f,pangolin::Attach::Pix(0),1.0f,(float)640/480)
		.SetLock(pangolin::LockLeft, pangolin::LockTop);
		
	pangolin::GlTexture imageTexture(640,480,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::OpenGlMatrixSpec P = pangolin::ProjectionMatrixRDF_TopLeft(640,480,535.4,539.2,320.1,247.6,0.001,1000);
	
	pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
	

	cv::namedWindow("rgb");
	cv::namedWindow("depth");
	Camera tum3;
    tum3.fx = 535.4;
    tum3.fy = 539.2;
    tum3.cx = 320.1;
    tum3.cy = 247.6;
    tum3.scale = 5000.0;
	
	PointCloud::Ptr globalMap ( new PointCloud() ); 
	PointCloud::Ptr tmp(new PointCloud());
	
	waitKey();
	
	vector<Eigen::Matrix4d> poses;
	//ifstream fin(dir+"/res/CameraTrajectory.txt");
	ifstream fin(dir+"jpl.txt");
	int count=0;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	while(!pangolin::ShouldQuit())
	{
		cout<<count<<endl;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		
		//d_cam.Activate(s_cam);
		d_image.Activate();
		glColor3f(1.0,1.0,1.0);
				
		while(!fin.eof())
		{
			count++;
			//cout<<count++<<endl;
			string s;
			getline(fin,s);
			//if(s[0]=='#')continue;
			if(!s.empty())
			{
				stringstream ss(s.c_str());
				double t,tx,ty,tz,qx,qy,qz,qw;
				double t1,t2;
				string s1,s2;
				fin>>t>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
				fin>>t1>>s1>>t2>>s2;
				Eigen::Matrix4d pose1=toTransformation(tx,ty,tz,qx,qy,qz,qw);
				
				Eigen::Matrix4d pose=pose1.inverse();
				
				std::vector<GLfloat> Twc = {
					(float)pose(0,0),(float)pose(1,0),(float)pose(2,0),0,
					(float)pose(0,1),(float)pose(1,1),(float)pose(2,1),0,
					(float)pose(0,2),(float)pose(1,2),(float)pose(2,2),0,
					(float)pose(0,3),(float)pose(1,3),(float)pose(2,3),1};
					
				pangolin::OpenGlMatrix M;

				M.m[0] = pose(0,0);
				M.m[1] = pose(1,0);
				M.m[2] = pose(2,0);
				M.m[3]  = 0.0;

				M.m[4] = pose(0,1);
				M.m[5] = pose(1,1);
				M.m[6] = pose(2,1);
				M.m[7]  = 0.0;

				M.m[8] = pose(0,2);
				M.m[9] = pose(1,2);
				M.m[10] = pose(2,2);
				M.m[11] = 0.0;

				M.m[12] = pose(0,3);
				M.m[13] = pose(1,3);
				M.m[14] = pose(2,3);
				M.m[15]  = 1.0;
					
				//poses.push_back(pose);
				Mat rgb=cv::imread(dir+s1, CV_LOAD_IMAGE_UNCHANGED);
				Mat depth=cv::imread(dir+s2,CV_LOAD_IMAGE_ANYDEPTH);

				imageTexture.Upload(rgb.data,GL_RGB,GL_UNSIGNED_BYTE);
				imageTexture.RenderToViewportFlipY();
				
				glClear(GL_DEPTH_BUFFER_BIT);

				// Load camera projection
				glMatrixMode(GL_PROJECTION);
				P.Load();
				
				glMatrixMode(GL_MODELVIEW);
				M.Load();
					
				
				//std::vector<GLfloat> Twc = {1,0,0,0, 0,1,0,0 , 0,0,1,0 ,2,0,0,1};//frame.mTwc;
				//glMultMatrixf(Twc.data());
				
			
// 	
// 				
// 				const float w = 0.02;
// 				const float h = w*0.75;
// 				const float z = w*0.6;
// 				glLineWidth(2);
// 				glColor3f(0.0,1.0,0.0);
// 				glBegin(GL_LINES);
// 				glVertex3f(0,0,0);
// 				glVertex3f(w,h,z);
// 				glVertex3f(0,0,0);
// 				glVertex3f(w,-h,z);
// 				glVertex3f(0,0,0);
// 				glVertex3f(-w,-h,z);
// 				glVertex3f(0,0,0);
// 				glVertex3f(-w,h,z);
// 				glVertex3f(w,h,z);
// 				glVertex3f(w,-h,z);
// 				glVertex3f(-w,h,z);
// 				glVertex3f(-w,-h,z);
// 				glVertex3f(-w,h,z);
// 				glVertex3f(w,h,z);
// 				glVertex3f(-w,-h,z);
// 				glVertex3f(w,-h,z);
// 				glEnd();
				
				{
					pangolin::OpenGlMatrix M = pangolin::OpenGlMatrix::Translate(-2,0,-0.1);
					glPushMatrix();
					M.Multiply();
					pangolin::glDrawColouredCube(-0.1,0.1);
					glPopMatrix();
					
// 					M = pangolin::OpenGlMatrix::Translate(-1.5,0.5,-0.1);
// 					glPushMatrix();
// 					M.Multiply();
// 					pangolin::glDrawColouredCube(-0.1,0.1);
// 					glPopMatrix();
				}
				
 				Mat rs=rgb.clone();
 				Frame frame(0,rgb,depth,tum3);
// 				{
// 					glLineWidth(10);
// 					glColor3f(1.0,0.0,0.0);
// 					glBegin(GL_LINES);
// 					for(int i=0 ;i< frame.lines.size(); i++)
// 					{
// 						cv::Point3d A = frame.lines[i].line3d.A;
// 						cv::Point3d B = frame.lines[i].line3d.B;
// 						
// 						if(cv::norm(A-B)<0.3)continue;
// 						
// 						glVertex3f(A.x,A.y,A.z);
// 						glVertex3f(B.x,B.y,B.z);
// 						
// 					}
// 					glEnd();
// 				}
// 				
// 				
// 				{
// 					glPointSize(1);
// 					glBegin(GL_POINTS);
// 					
// 					PointCloud::Ptr cloud = img2cloud(rgb,depth,tum3);
// 					for(int k=0; k<cloud->points.size(); k++)
// 					{
// 						PointT p = cloud->points[k];
// 						
// 						uint32_t x = p.rgba;
// 						
// 						uint32_t alpha = (x & 0xff000000) >> 24;
// 						uint32_t red = (x & 0x00ff0000) >> 16;
// 						uint32_t green = (x & 0x0000ff00) >> 8;
// 						uint32_t blue = (x & 0x000000ff);
// 						
// 						glColor3f(red/255.0,green/255.0,blue/255.0);
// 						glVertex3f(p.x,p.y,p.z);
// 					}
// 					glEnd();
// 					cloud->clear();
// 				}
				
				
				for(int k=0; k<frame.lines.size();k++)
				{
					cv::Point2f left1 = frame.lines[k].p;
					cv::Point2f left2 = frame.lines[k].q;
					cv::line(rs,left1,left2,cv::Scalar(0,255,0),3);
				}
				for(int k=0; k<frame.mvKeypoints.size();k++)
				{
					circle(rs,frame.mvKeypoints[k].pt,1,Scalar(0,0,255),2);
				}
				//frame.planeExtraction();
			
				
				cv::imshow("rgb",rs);
				cv::imshow("depth",depth);
				cv::waitKey(50);
				
				//glPopMatrix();
				break;
			}
			
		}
		
		pangolin::FinishFrame();
		
	}
	//pcl::io::savePCDFile("global.pcd",*globalMap);
	fin.close();
	
	return 0;
}


PointCloud::Ptr img2cloud_test(Mat rgb, Mat depth, Camera camera)
{
    PointCloud::Ptr cloud(new PointCloud);
    for(int i=0;i<depth.rows;i+=3)
    {
		for(int j=0;j<depth.cols;j+=3)
		{
			double d=(uchar)depth.ptr<uchar>(i)[j]/camera.scale;
			//cout<<d<<endl;
			if(d <= 1||d>=10)continue;
			PointT p;
			p.z= d;
			p.x=(j-camera.cx)*p.z/camera.fx;
			p.y=(i-camera.cy)*p.z/camera.fy;
			
			p.b=rgb.ptr<uchar>(i)[j*3];
			p.g=rgb.ptr<uchar>(i)[j*3+1];
			p.r=rgb.ptr<uchar>(i)[j*3+2];
			cloud->points.push_back(p);
		}
    }
   
    cloud->width=cloud->points.size();
    cloud->height=1;
    cloud->is_dense=false;
    
    return cloud;
}

int drawPointCloud(int idx)
{
	//int idx=2;
	string dir = "/home/jpl/programming/zjh/data"+to_string(idx)+"/";
	Mat rgb=cv::imread(dir+"im1.png", 	CV_LOAD_IMAGE_UNCHANGED);
	Mat depth=cv::imread(dir+"fuse.png",CV_LOAD_IMAGE_ANYDEPTH);
	//cv::imshow("rgb",rgb);
    //cv::imshow("depth",depth);
	//waitKey();
	
	Camera tum[4];
    tum[1].fx = 713.189;
    tum[1].fy = 713.189;
    tum[1].cx = 387.361;
    tum[1].cy = 238.263;
    tum[1].scale = 25.5;
	
    tum[2].fx = 1500;
    tum[2].fy = 1500;
    tum[2].cx = 251;
    tum[2].cy = 187;
    tum[2].scale = 25.5;
	
    tum[3].fx = 585.489;
    tum[3].fy = 585.489;
    tum[3].cx = 312.353;
    tum[3].cy = 234.124;
    tum[3].scale = 25.5;
	
    tum[4].fx = 1038.018;
    tum[4].fy = 1038.018;
    tum[4].cx = 375.308;
    tum[4].cy = 243.393;
    tum[4].scale = 25.5;
	
	pcl::visualization::CloudViewer viewer("viewer");
	PointCloud::Ptr cloud = img2cloud_test(rgb,depth,tum[idx]);
	
	
// 	pcl::VoxelGrid<PointT> voxel;
// 	double gridsize = 0.01; 
// 	voxel.setLeafSize( gridsize, gridsize, gridsize );
// 	voxel.setInputCloud( cloud );
// 	voxel.filter( *cloud );
	
	
	viewer.showCloud(cloud);
	//viewer.runOnVisualizationThread(setPCLBackGround);
    while(!viewer.wasStopped()){}
	
	return 0;
}

int main(int argc, char** argv)
{
	//reconstruction_pangolin("/home/jpl/TUM_Datasets/4Structure_vs_Texture/f3_structure_notexture_far/");
	//reconstruction("/home/jpl/TUM_Datasets/6Reconstruction/f3_large_cabinet/");
	//testline();
	//test2(argv[1],argv[2],atoi(argv[3]));
	//test();
	//testDBoW();
	int idx=atoi(argv[1]);
	drawPointCloud(idx);
	
	return 0;
}
