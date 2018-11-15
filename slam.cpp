#include "base.h"
#include "utils.h"
#include "time.h"
#include <sys/time.h>
#include <unistd.h>
#include "SysParams.h"
#include "PnPsolver.h"
#include <opencv2/core/eigen.hpp>

typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	

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
      
      //Eigen::Quaterniond q(r);
      //cout<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;
      
      return T;
}

double normTransform(Mat rvec,Mat tvec)
{
  
      //
      return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

/*
PointCloud::Ptr joinPointCloud(PointCloud::Ptr cloud,Frame frame,Eigen::Isometry3d& T)
{
      PointCloud::Ptr newcloud=frame.img2cloud();
      PointCloud::Ptr output(new PointCloud);
      pcl::transformPointCloud(*cloud,*output,T.matrix());
      *newcloud+=*output;
      return newcloud;
}
*/


bool checkKeyframe(Frame& keyFrame,Frame& frame,g2o::SparseOptimizer& opti)
{
	bool isKeyframe=true;
	const int min_inliers=5;
	const double max_norm = 0.6;
	const double keyframe_threshold = 0.2;
	const double max_norm_lp =5;

	g2o::RobustKernel* robustKernel=g2o::RobustKernelFactory::instance()->construct("Cauchy");

	PnPsolver pnpsolver;
	pnpsolver.calculateParameter(keyFrame,frame);
	if(pnpsolver.inliers.rows < min_inliers)return false;

	double norm = normTransform(pnpsolver.rvec, pnpsolver.tvec);
	cout<<"Norm: "<<norm<<endl;

	//if (norm>=max_norm) return false;
	if(norm<=keyframe_threshold && pnpsolver.inliers.rows > 20) return false;
	
	cout<<"Insert new keyframe "<<frame.id<<" "<<endl;

	g2o::VertexSE3 *v = new g2o::VertexSE3();
	v->setId(frame.id);
	v->setEstimate(Eigen::Isometry3d::Identity());
	opti.addVertex(v);
	
	
	g2o::EdgeSE3* edge=new g2o::EdgeSE3();
	edge->vertices()[0]= opti.vertex(keyFrame.id);
	edge->vertices()[1]= opti.vertex(frame.id);
	edge->setRobustKernel(robustKernel);
	//information matrix
	Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Identity();
	information(0,0)=information(1,1)=information(2,2)=100;
	information(3,3)=information(4,4)=information(5,5)=100;
	edge->setInformation(information);

	Eigen::Isometry3d T=cvMat2Eigen(pnpsolver.rvec,pnpsolver.tvec);
	edge->setMeasurement(T.inverse());
	opti.addEdge(edge);
	return true;
}


void loadImage(string rootpath,const string strAssociationFilename,  vector<double>& vdTimestamps,
	       vector<string>& vstrFilenamesRGB,  vector<string>& vstrFilenamesDepth)
{
  //string root = "/home/jpl/TUM_Datasets/3Robot/f2_pioneer_slam/";
  //string root = "/home/jpl/TUM_Datasets/4Structure_vs_Texture/f3_structure_notexture_near/";
 string root = rootpath;
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


int main(int agrc, char** argv)
{
    
    struct timeval tstart,tend;
    //SysParams sysparams("/home/jpl/lines/TUM3.yaml");
    
	string rootpath=argv[1];
	SysParams sysparams(argv[2]);
	
	Camera camera(sysparams);
    //string strAssociationFilename = "/home/jpl/TUM_Datasets/3Robot/f2_pioneer_slam/associations.txt";
    string strAssociationFilename = rootpath + "associations.txt";//= "/home/jpl/TUM_Datasets/4Structure_vs_Texture/f3_structure_notexture_near/associations.txt";
 
    int nImages = 0;
    vector<double> vdTimestamps;
    vector<string> vstrFilenamesRGB;
    vector<string> vstrFilenamesDepth;
    //load the filename of rgb and depth
    loadImage(rootpath, strAssociationFilename,vdTimestamps, vstrFilenamesRGB, vstrFilenamesDepth);
    nImages = vstrFilenamesRGB.size();
    if(vstrFilenamesRGB.empty())return 0;
    if(vstrFilenamesRGB.size()!=vstrFilenamesDepth.size())return 0;
    
    
    gettimeofday(&tstart,NULL);
    
    Mat rgb1 = imread(vstrFilenamesRGB[0], CV_LOAD_IMAGE_UNCHANGED );
    Mat depth1 = imread(vstrFilenamesDepth[0], CV_LOAD_IMAGE_ANYDEPTH);
    Frame frame1(vdTimestamps[0],rgb1,depth1,camera);

    vector<Frame> allFrame;
    vector<Frame> keyFrame;
    allFrame.push_back(frame1);
    keyFrame.push_back(frame1);     
    

    /*g2o*/
    SlamLinearSolver* linearSolver=new SlamLinearSolver();
    SlamBlockSolver* blockSolver=new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* algorithm=new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    
    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm(algorithm);
    globalOptimizer.setVerbose(false);//without debug information

    g2o::VertexSE3* v=new g2o::VertexSE3();
    v->setId(0);
    v->setEstimate(Eigen::Isometry3d::Identity()); 
    v->setFixed(true);
    globalOptimizer.addVertex(v);

    nImages = 400;
    
    for(size_t i=1;i <nImages;i++)  //nImages
    {
		cout<<i<<endl;
		Mat rgb2=imread(vstrFilenamesRGB[i],CV_LOAD_IMAGE_UNCHANGED);
		Mat depth2=imread(vstrFilenamesDepth[i],CV_LOAD_IMAGE_UNCHANGED);
		Frame frame2(vdTimestamps[i],rgb2,depth2,camera);

		bool isKeyframe=checkKeyframe(keyFrame.back(),frame2,globalOptimizer);
		if(isKeyframe){
		keyFrame.push_back(frame2);
		}
		allFrame.push_back(frame2);
		//fprintf(stderr,"\rFinish %5.2f%% ",(double)i*100/num);
    } 
    //optimization
    cout<<"optimizing pose graph. Vertices "<<globalOptimizer.vertices().size()<<endl;
    //globalOptimizer.save("result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);
    //globalOptimizer.save("result_after.g2o");
    cout<<"optimization done."<<endl;
    
    gettimeofday(&tend, NULL);
    double timeUsed = 1000000*(tend.tv_sec-tstart.tv_sec)+tend.tv_usec-tstart.tv_usec;
    cout<<nImages/(timeUsed/1e6)<<" fps"<<endl;
      

    PointCloud::Ptr globalMap ( new PointCloud() ); 
    PointCloud::Ptr tmp(new PointCloud());
    
    pcl::VoxelGrid<PointT> voxel;
    double gridsize = 0.01; 
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    
    
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

    for (size_t i=0; i<keyFrame.size(); i++)
    {       
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyFrame[i].id ));
        Eigen::Isometry3d pose = vertex->estimate(); 
	
        PointCloud::Ptr p = keyFrame[i].img2cloud();
        pcl::transformPointCloud( *p, *tmp, pose.matrix() );
        *globalMap += *tmp;
        tmp->clear();
        p->clear();
    }
    

    voxel.setInputCloud( globalMap );
    voxel.filter( *tmp );
    globalMap.swap(tmp);
    cout<<"Global map size："<<globalMap->points.size()<<endl;
    
    pcl::visualization::CloudViewer viewer("viewer");
    viewer.showCloud(globalMap);
    while(!viewer.wasStopped()){}

    globalOptimizer.clear();
    

    return 0;
}
