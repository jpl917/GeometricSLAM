#include <pangolin/pangolin.h>
#include "base.h"
#include "utils.h"
#include "time.h"
#include <sys/time.h>
#include <unistd.h>
#include "SysParams.h"
#include "PnPsolver.h"
#include "Viewer.h"
#include "pydensecrf/pydensecrf/densecrf/include/Eigen/src/Core/products/GeneralBlockPanelKernel.h"
#include <iostream>

#include <opencv2/core/eigen.hpp>

typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	
//#define DISABLE_POINT
//#define DISABLE_LINE

bool tooFar(Eigen::Matrix4d m)
{
	//if(m(0,0)<0.8||m(1,1)<0.8||m(2,2)<0.8||m(0,0)==1||m(1,1)==1||m(2,2)==1
	//	  ||m(0,3)*m(0,3)+m(1,3)*m(1,3)+m(2,3)*m(2,3)>1)
		
      if(m(0,0)<0.95||m(1,1)<0.95||m(2,2)<0.95||m(0,0)==1||m(1,1)==1||m(2,2)==1
	    ||m(0,3)*m(0,3)+m(1,3)*m(1,3)+m(2,3)*m(2,3)>0.09)
	  {
		  cout<<"Look"<<m<<endl;
		  return true;
	  }
	  else return false;
}


void drawPangolin(vector<Frame>& frames);
void testSeg(PointCloud::Ptr cloud);

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

Eigen::Isometry3d cvMat2Eigen(Mat& rvec,Mat& tvec)
{
      Mat R;
      cv::Rodrigues(rvec,R);
      Eigen::Matrix3d r;
      cv::cv2eigen(R,r);

      Eigen::Isometry3d T=Eigen::Isometry3d::Identity(); 
      Eigen::AngleAxisd angle(r);
      T=angle;

      T(0,3)=tvec.at<double>(0,0);
      T(1,3)=tvec.at<double>(1,0);
      T(2,3)=tvec.at<double>(2,0);
      
      return T;
}

double normTransform(Mat rvec,Mat tvec)
{
      return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}


void isKeyframe(Frame& keyFrame,Frame& frame,g2o::SparseOptimizer& opti, Eigen::Isometry3d T)
{
	g2o::VertexSE3 *v = new g2o::VertexSE3();
	v->setId(frame.id);
	v->setEstimate(Eigen::Isometry3d::Identity());
	opti.addVertex(v);
	
	g2o::EdgeSE3* edge=new g2o::EdgeSE3();
	edge->vertices()[0]= opti.vertex(keyFrame.id);
	edge->vertices()[1]= opti.vertex(frame.id);
	g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
	rk->setDelta(5.99);
	edge->setRobustKernel(rk);
	//information matrix
	Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Identity();
	information(0,0)=information(1,1)=information(2,2)=100;
	information(3,3)=information(4,4)=information(5,5)=100;
	edge->setInformation(information);
	
	edge->setMeasurement(T);
	opti.addEdge(edge);
	return;
}


void loadImage(string rootpath, const string strAssociationFilename,  vector<double>& vdTimestamps,
	       vector<string>& vstrFilenamesRGB,  vector<string>& vstrFilenamesDepth)
{
  
	string root = rootpath;//"/home/jpl/TUM_Datasets/1Testing/f1_xyz/";
	
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
			vstrFilenamesRGB.push_back(root+sRgb);
			vstrFilenamesDepth.push_back(root+sDepth);
		}
	}
	return;
}

void saveTUMAllTrajectory(vector<Frame> AllFrame)
{
	
}

void saveTUMKeyTrajectory(vector<Frame> keyFrame)
{
	//sort(keyFrame.begin(),keyFrame.end(),Frame::id);
    ofstream out("traject.txt");
    out<<fixed;
    for(size_t i=0; i<keyFrame.size();i++)
    {
		cv::Mat Tcw = keyFrame[i].mTcw;
		cv::Mat Rwc = keyFrame[i].mRwc;
		cv::Mat twc = keyFrame[i].mOw;
		
		Eigen::Matrix3d r;
		cv::cv2eigen(Rwc,r);
		Eigen::Quaterniond q(r);
		out<<setprecision(6)<<keyFrame[i].timestamp<<" "<<setprecision(9)<<twc.at<double>(0)<<" "<<twc.at<double>(1)<<" "<<twc.at<double>(2)<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;
    }
    out.close();
}


int main(int argc, char** argv)
{	
	if(argc!=3){
		cout<<"No parameters."<<endl;
		return 0;
	}
	string rootpath=argv[1];
	SysParams sysparams(argv[2]); 
	
	SystemParameters sysPara;   //line parameters
	Camera camera(sysparams);
	
	int window_length_keyframe = 10;  //10

	string strAssociationFilename = rootpath + "associations.txt";
	//string strAssociationFilename = rootpath + "syncidx.txt";
	
	int nImages = 0;
	vector<double> vdTimestamps;
	vector<string> vstrFilenamesRGB, vstrFilenamesDepth;
	//load the filename of rgb and depth
	loadImage(rootpath, strAssociationFilename,vdTimestamps, vstrFilenamesRGB, vstrFilenamesDepth);
	nImages = vstrFilenamesRGB.size();
	if(vstrFilenamesRGB.empty())return 0;
	if(vstrFilenamesRGB.size()!=vstrFilenamesDepth.size())return 0;

	nImages = 300;  //Number of images. For testing
	
	//First Frame
	Mat rgb1 = imread(vstrFilenamesRGB[0], CV_LOAD_IMAGE_UNCHANGED );
	Mat depth1 = imread(vstrFilenamesDepth[0], CV_LOAD_IMAGE_ANYDEPTH);
	Frame frame1(vdTimestamps[0],rgb1,depth1,camera);
	frame1.rgbname=vstrFilenamesRGB[0];

	vector<Frame> allFrame;
	vector<Frame> keyFrame;
	allFrame.push_back(frame1);
	keyFrame.push_back(frame1);     

	//for global optimization
	SlamLinearSolver* linearSolver = new SlamLinearSolver();
	SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    
	g2o::SparseOptimizer globalOptimizer;
	globalOptimizer.setAlgorithm(algorithm);
	
	g2o::VertexSE3* v = new g2o::VertexSE3();
	v->setId(0);
	v->setEstimate(Eigen::Isometry3d::Identity());
	v->setFixed(true);
	globalOptimizer.addVertex(v);
	
	//for global map
	PointCloud::Ptr globalMap ( new PointCloud() ); 
	PointCloud::Ptr tmp(new PointCloud());
	pcl::VoxelGrid<PointT> voxel;
	double gridsize = 0.03; 
	voxel.setLeafSize( gridsize, gridsize, gridsize );

	MyTimer mytimer;
	mytimer.start();
    for(int i=1; i < nImages; i+=1)  //nImages
    {
		int keyFrameflag=keyFrame.size();
		for(int k=0; k<1;k++)
		{
			//frame1 = keyFrame.back();
			if(keyFrameflag-1-10*k<0)break;
			frame1 = keyFrame[keyFrameflag-1-10*k];
			
			cout<<"---------------------------------------------------------------"<<endl;
			cout<<"Frame id:"<<i<<endl;
			Mat rgb2=imread(vstrFilenamesRGB[i],CV_LOAD_IMAGE_UNCHANGED);
			Mat depth2=imread(vstrFilenamesDepth[i],CV_LOAD_IMAGE_UNCHANGED);
			Frame frame2(vdTimestamps[i],rgb2,depth2,camera);
			frame2.rgbname = vstrFilenamesRGB[i];
			
			frame2.AHCPlane();
			//continue;

			//brightness
			double bright = ave_img_bright(frame2.rgb);
			if(sysPara.max_img_brightness < bright) sysPara.max_img_brightness = bright;
			if(bright/sysPara.max_img_brightness<0.3 || bright < 10)
			{
				cout<<"Dark"<<endl;
				sysPara.dark_lighting = true;
				sysPara.pt2line3d_dist_relmotion = 0.025; //m
				sysPara.line3d_angle_relmotion = 5;
				sysPara.fast_motion = 0;
			}
			else{
				cout<<"Bright"<<endl;
				sysPara.dark_lighting = false;
				sysPara.pt2line3d_dist_relmotion = 0.05; //m
				sysPara.line3d_angle_relmotion = 10;
				sysPara.fast_motion = 1;
			}
			
			/**************************************************/
			/*****************Point match**********************/
			/**************************************************/
			vector<DMatch> pt_matches;
			vector<DMatch> bf_matches;
			BruteForceMatcher<HammingLUT> bf_matcher;
			if(frame1.mDescriptors.rows!=0 && frame2.mDescriptors.rows!= 0)
			{
				bf_matcher.match(frame1.mDescriptors,frame2.mDescriptors,bf_matches);

#ifdef GMS_MATCHER
				pt_matches = GmsMatch(frame1,frame2, bf_matches);
#else
				pt_matches = FilterMatch(frame1,frame2,bf_matches);
#endif
			}
			
			
			/**************************************************/
			/****************Line match**********************/
			/**************************************************/
			vector<vector<int>> matches;
			vector<DMatch>      ln_matches;
			
			trackLine(frame1.lines,frame2.lines, matches, sysPara);
			for(size_t j=0; j<matches.size(); j++)
			{
				int p1 = matches[j][0];
				int p2 = matches[j][1];
				if(!frame1.lines[p1].haveDepth || !frame2.lines[p2].haveDepth)continue;
				double len1 = cv::norm(frame1.lines[p1].line3d.A - frame1.lines[p1].line3d.B);
				double len2 = cv::norm(frame2.lines[p2].line3d.A - frame2.lines[p2].line3d.B);
				DMatch tmp(matches[j][0],matches[j][1],0);
				ln_matches.push_back(tmp);
			}
			
			cout<<"F1 Lines:"<<frame1.lines.size()<<"  F2 Lines: "<<frame2.lines.size()<<"  Line Matches: "<<ln_matches.size()<<endl;
			
			/**************************************************/
			/****************Feature visulalization************/
			/**************************************************/
			Mat img;
			drawMatches(frame1.rgb,frame1.mvKeypoints,frame2.rgb,frame2.mvKeypoints,pt_matches,img);
			cv::imshow("Point Match",img);
			//if(pt_matches.size()<5&&frame2.id<200)waitKey(500);
			cvWaitKey(1);
			
			cv::Mat output = drawInlier(frame1,frame2, matches);
			cv::Mat src1,src2;
			frame1.rgb.copyTo(src1);
			frame2.rgb.copyTo(src2);
			srand((unsigned int)time(NULL));
			for(size_t k=0; k<matches.size();k++)
			{
				int p = matches[k][0];
				int q = matches[k][1];
				cv::Point2f left1 = frame1.lines[p].p;
				cv::Point2f left2 = frame1.lines[p].q;
				cv::Point2f right1 = frame2.lines[q].p + cv::Point2d((double)src1.cols,0.f);
				cv::Point2f right2 = frame2.lines[q].q + cv::Point2d((double)src1.cols,0.f);
				cv::line(output,left1,left2,cv::Scalar(0,0,255),2);
				cv::line(output,right1,right2,cv::Scalar(0,0,255),2);
				int r = rand()%255;
				int g = rand()%255;
				int b = rand()%255;
				
				cv::Point2f left = cv::Point2f((left1.x+left2.x)/2,(left1.y+left2.y)/2);
				cv::Point2f right = cv::Point2f((right1.x+right2.x)/2,(right1.y+right2.y)/2);
				cv::line(output,left,right,cv::Scalar(b,g,r),2);
			}
			cv::imshow("Line Match", output);
			cvWaitKey();
			
			
			int line_threshold = std::max(3.0, 0.3* 0.5* (frame1.lines.size() + frame2.lines.size()));
			//Decide keyframes
			if(frame2.id-frame1.id >= window_length_keyframe) //line_threshold
			//if(ln_matches.size() < line_threshold ||frame2.id-frame1.id >= window_length_keyframe) //line_threshold
			{

#ifdef DISABLE_POINT
				{
					vector<DMatch> t;
					pt_matches = t;
				}
#endif
#ifdef DISABLE_LINE
				{
					vector<DMatch> t;
					ln_matches = t;
				}
#endif
				vector<DMatch> t;
				if(ln_matches.size() == 0&&pt_matches.size()==0)continue;
				if(pt_matches.size()<50)  pt_matches = t;
				//if(pt_matches.size()>10*ln_matches.size()) ln_matches=t;
				
				cout<<"Insert Keyframe:"<<frame2.id<<endl;
				cout<<"Pt matches: "<<pt_matches.size()<<"      Ln matches: "<<ln_matches.size()<<endl;
				
				bool valid;
				//Eigen::Matrix4f T = getTransform_Lns_Pts_pcl(&frame1,&frame2,pt_matches,ln_matches,valid);
				Eigen::Matrix4f Tt = Eigen::Matrix4f::Identity();
				Tt = getIcpAlignment( &frame1,  &frame2);
				Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
				T = Tt.inverse();
				cout<<"T-before"<<T<<endl;
				
#ifdef SEGMENT
				if(!frame1.segOK)
					frame1.segment();
				if(!frame2.segOK)
					frame2.segment();
#endif
				
				Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();
// 				if(pt_matches.size()==0)  //only lines 
// 				{					
// 					T1 = getTransformFromHybridMatchesG2O(&frame1,&frame2,pt_matches,ln_matches,T,10,sysPara);
// 					//bool valid;
// 					//T1 = getTransform_Line_svd(&frame1, &frame2, ln_matches,valid);	
// 				}
// 				else
// 				{
// 					vector<DMatch> out_pt_matches, out_ln_matches;
// 					float inlier_rmse;
// 					//T1 = getTransform_PtsLines_ransac(&frame1,&frame2,pt_matches,ln_matches,out_pt_matches,out_ln_matches,T, inlier_rmse, sysPara);
// 					T1 = getTransformFromHybridMatchesG2O(&frame1,&frame2,pt_matches,ln_matches,T,10,sysPara);
// 				}
				
				
				/*****************test plane alignment***********************/
				Eigen::Matrix3d rot;
				rot << T(0,0), T(0,1), T(0,2),
					   T(1,0), T(1,1), T(1,2),
					   T(2,0), T(2,1), T(2,2);
				
				Eigen::AngleAxisd v;
				v.fromRotationMatrix(rot);
				
				T1.rotate(v);
				T1.pretranslate(Eigen::Vector3d(T(0,3),T(1,3),T(2,3)));
				/****************end test*************************************/
							
				
				
				if(tooFar(T1.matrix()))
				{
					cout<<"Too Far"<<endl;
					//cv::waitKey();
					//continue;
				}
				cout<<T1.matrix()<<endl;

				
				if(k==0) keyFrame.push_back(frame2);
				isKeyframe(frame1,frame2,globalOptimizer,T1);
				
			}
		}
    } 
    mytimer.end();

	/*
	g2o::EdgeSE3* edge=new g2o::EdgeSE3();
	edge->vertices()[0]= globalOptimizer.vertex(0);
	edge->vertices()[1]= globalOptimizer.vertex(keyFrame.back().id);
	g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
	rk->setDelta(5.99);
	edge->setRobustKernel(rk);
	//information matrix
	Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Identity();
	information(0,0)=information(1,1)=information(2,2)=1;
	information(3,3)=information(4,4)=information(5,5)=1;
	edge->setInformation(information);
	Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
	edge->setMeasurement(T2);
	globalOptimizer.addEdge(edge);*/
	
#ifdef GLOBAL_BA
    //optimization
    globalOptimizer.setVerbose(true);  
    cout<<"optimizing pose graph. Vertices "<<globalOptimizer.vertices().size()<<endl;
    //globalOptimizer.save("result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);
    //globalOptimizer.save("result_after.g2o");
    cout<<"optimization done."<<endl;
	
	
	//save trajectory
	for (size_t i=0; i<keyFrame.size(); i++)
    { 
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyFrame[i].id ));
        Eigen::Isometry3d pose = vertex->estimate(); 
		
		keyFrame[i].setPose(toCvMat(pose.matrix()).inv());
			
        PointCloud::Ptr p = keyFrame[i].img2cloud();  //keyFrame[i].planeExtraction();
        pcl::transformPointCloud( *p, *tmp, pose.matrix()); //pose = Twc
        *globalMap += *tmp;
        tmp->clear();
        p->clear();
    }
    

    
    voxel.setInputCloud( globalMap );
    voxel.filter( *tmp );
    globalMap.swap(tmp);
    cout<<"KeyFrame size:  "<<keyFrame.size()<<endl;
    cout<<"Global map size："<<globalMap->points.size()<<endl;
    //pcl::io::savePCDFileASCII("1.pcd",*globalMap);
#endif
	
	//drawPangolin(keyFrame);
	
	saveTUMKeyTrajectory(keyFrame);
	pcl::io::savePCDFile("global.pcd",*globalMap);
	
	testSeg(globalMap);
	
    pcl::visualization::CloudViewer viewer("viewer");
    viewer.showCloud(globalMap);
	//viewer.runOnVisualizationThread(setPCLBackGround);
    while(!viewer.wasStopped()){}

    return 0;
}

void testSeg(PointCloud::Ptr cloud)
{
	pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree = 
		boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA> > (new pcl::search::KdTree<pcl::PointXYZRGBA>);
	pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod (tree);
	normal_estimator.setInputCloud (cloud);
	normal_estimator.setKSearch (100);
	//normal_estimator.setRadiusSearch (0.05);
	normal_estimator.compute (*normals);

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

	std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
	vector<double> c;
	srand(unsigned(time(NULL)));
	//Mat rs= frame.rgb.clone();
	for(int i=0; i<clusters.size(); i++)
	{
		//std::cout << "Cluster " <<i<<" "<<clusters[i].indices.size () << " points." << endl;
		//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr each_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
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
			//each_plane->push_back(point);
			cloud_plane->push_back(point);
		}
		//getEquation(each_plane,c);
	}
	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
	pcl::visualization::CloudViewer viewer ("Cluster viewer");
	viewer.showCloud(cloud_plane);
	while (!viewer.wasStopped ()){}
	
	/**/
	//return cloud_plane;
}

void drawPangolin(vector<Frame>& frames)
{
	
	float mImageWidth = 640;
	float mImageHeight = 480;
	float mViewpointX = 0;
	float mViewpointY = -0.7;
	float mViewpointZ = -1.8;
	float mViewpointF = 500;
	
	pangolin::CreateWindowAndBind("LineSLAM",1024,768);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	
	pangolin::OpenGlRenderState s_cam(
			pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
			pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,1.0, 0.0)
			);
			
	pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
	
	pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
	
	while(!pangolin::ShouldQuit())
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);
		
		//pangolin::glDrawAxis(3);
		//pangolin::glDrawColouredCube();
		
		
		for(int idx = 0; idx<frames.size(); idx+=1)
		{
			
			Mat mRwc = frames[idx].mRcw;
			Mat mOw = frames[idx].mOw;
			
			
			std::vector<GLfloat> Twc = {
				(float)mRwc.at<double>(0,0),(float)mRwc.at<double>(0,1),(float)mRwc.at<double>(0,2),0,
				(float)mRwc.at<double>(1,0),(float)mRwc.at<double>(1,1),(float)mRwc.at<double>(1,2),0,
				(float)mRwc.at<double>(2,0),(float)mRwc.at<double>(2,1),(float)mRwc.at<double>(2,2),0,
				(float)mOw.at<double>(0),(float)mOw.at<double>(1),(float)mOw.at<double>(2),1
			};//frame.mTwc;
			
			glPushMatrix();
			//std::vector<GLfloat> Twc = {1,0,0,0, 0,1,0,0 , 0,0,1,0 ,2,0,0,1};//frame.mTwc;
			glMultMatrixf(Twc.data());
			
			const float w = 0.02;
			const float h = w*0.75;
			const float z = w*0.6;
			glLineWidth(2);
			glColor3f(0.0,1.0,0.0);
			glBegin(GL_LINES);
			glVertex3f(0,0,0);
			glVertex3f(w,h,z);
			glVertex3f(0,0,0);
			glVertex3f(w,-h,z);
			glVertex3f(0,0,0);
			glVertex3f(-w,-h,z);
			glVertex3f(0,0,0);
			glVertex3f(-w,h,z);
			glVertex3f(w,h,z);
			glVertex3f(w,-h,z);
			glVertex3f(-w,h,z);
			glVertex3f(-w,-h,z);
			glVertex3f(-w,h,z);
			glVertex3f(w,h,z);
			glVertex3f(-w,-h,z);
			glVertex3f(w,-h,z);
			glEnd();
			
			
			if(idx<30)
			{
				glLineWidth(10);
				glColor3f(1.0,0.0,0.0);
				glBegin(GL_LINES);
				for(int i=0 ;i< frames[idx].lines.size(); i++)
				{
					cv::Point3d A = frames[idx].lines[i].line3d.A;
					cv::Point3d B = frames[idx].lines[i].line3d.B;
					
					if(cv::norm(A-B)<0.3)continue;
					
					glVertex3f(A.x,A.y,A.z);
					glVertex3f(B.x,B.y,B.z);
					
				}
				glEnd();
			}
			{
	
			
				glPointSize(1);
				glBegin(GL_POINTS);
				
				PointCloud::Ptr cloud = frames[idx].img2cloud();
				for(int k=0; k<cloud->points.size(); k++)
				{
					PointT p = cloud->points[k];
					
					uint32_t x = p.rgba;
					
					uint32_t alpha = (x & 0xff000000) >> 24;
					uint32_t red = (x & 0x00ff0000) >> 16;
					uint32_t green = (x & 0x0000ff00) >> 8;
					uint32_t blue = (x & 0x000000ff);
					
					glColor3f(red/255.0,green/255.0,blue/255.0);
					glVertex3f(p.x,p.y,p.z);
				}
				glEnd();
			}
			
			glPopMatrix();
			
		}
		
		pangolin::FinishFrame();
		
	}

	

}



















