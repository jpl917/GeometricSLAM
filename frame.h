#ifndef FRAME_H
#define FRAME_H

#include <Python.h>
#include "ORBextractor.h"
#include "Camera.h"
#include "base.h"
#include <numpy/ndarrayobject.h>
#include "AHCPlaneFitter.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

#define EPS (1e-10)
#define PI (3.1415926535)
#define EXTRACTLINE_USE_MAHDIST
#define USE_LINE
#define SLAM_LBA
#define LINEDIST_6D
//#define UNDISTORT

#define GLOBAL_BA

#define GMS_MATCHER
//#define SEGMENT

struct OrganizedImage3D {
    const cv::Mat_<cv::Vec3f>& cloud;
    //note: ahc::PlaneFitter assumes mm as unit!!!
    OrganizedImage3D(const cv::Mat_<cv::Vec3f>& c): cloud(c) {}
    inline int width() const { return cloud.cols; }
    inline int height() const { return cloud.rows; }
    inline bool get(const int row, const int col, double& x, double& y, double& z) const {
        const cv::Vec3f& p = cloud.at<cv::Vec3f>(row,col);
        x = p[0];
        y = p[1];
        z = p[2];
        return z > 0 && std::isnan(z)==0; //return false if current depth is NaN
    }
};
typedef ahc::PlaneFitter< OrganizedImage3D > PlaneFitter;

class SystemParameters;


class RandomPoint3d 
{
public:
	cv::Point3d 	pos;	
	cv::Mat			cov;
	cv::Mat			U, W; // cov = U*D*U.t, D = diag(W); W is vector
	double      	W_sqrt[3]; // used for mah-dist from pt to ln
	double 			DU[9];
	double      	dux[3];

	RandomPoint3d(){}
	RandomPoint3d(cv::Point3d _pos) 
	{
		pos = _pos;
		cov = cv::Mat::eye(3,3,CV_64F);
		U = cv::Mat::eye(3,3,CV_64F);
		W = cv::Mat::ones(3,1,CV_64F);
	}
	RandomPoint3d(cv::Point3d _pos, cv::Mat _cov)
	{
		pos = _pos;
		cov = _cov.clone();
		cv::SVD svd(cov);
		U = svd.u.clone();
		W = svd.w.clone();
		W_sqrt[0] = sqrt(svd.w.at<double>(0));
		W_sqrt[1] = sqrt(svd.w.at<double>(1));
		W_sqrt[2] = sqrt(svd.w.at<double>(2));

		cv::Mat D = (cv::Mat_<double>(3,3)<<1/W_sqrt[0], 0, 0, 
			0, 1/W_sqrt[1], 0,
			0, 0, 1/W_sqrt[2]);
		cv::Mat du = D*U.t();
		DU[0] = du.at<double>(0,0); 
		DU[1] = du.at<double>(0,1);
		DU[2] = du.at<double>(0,2);
		DU[3] = du.at<double>(1,0); 
		DU[4] = du.at<double>(1,1);
		DU[5] = du.at<double>(1,2);
		DU[6] = du.at<double>(2,0); 
		DU[7] = du.at<double>(2,1); 
		DU[8] = du.at<double>(2,2);
		dux[0] = DU[0]*pos.x + DU[1]*pos.y + DU[2]*pos.z;
		dux[1] = DU[3]*pos.x + DU[4]*pos.y + DU[5]*pos.z;
		dux[2] = DU[6]*pos.x + DU[7]*pos.y + DU[8]*pos.z;

	}
};


class RandomLine3d 
{
public:
	vector<RandomPoint3d> 	pts;  //supporting collinear points
	cv::Point3d 			A, B;
	cv::Mat 				covA, covB;
	RandomPoint3d 			rndA, rndB;
	cv::Point3d 			u, d; // following the representation of Zhang's paper 'determining motion from...'
	RandomLine3d () {}
	RandomLine3d (cv::Point3d _A, cv::Point3d _B, cv::Mat _covA, cv::Mat _covB) 
	{
		A = _A;
		B = _B;
		covA = _covA.clone();
		covB = _covB.clone();
	}
};

/*
class LmkLine
{
public:
	cv::Point3d			A, B;
	int					gid;
	vector<vector<int>> frmId_lnLid;
	LmkLine(){}
};*/


class Data_MLEstimateLine3d
{
public:
	int idx1, idx2;
	vector<RandomPoint3d> pts;
	Eigen::Matrix3d cov_inv_idx1, cov_inv_idx2;
	Data_MLEstimateLine3d(vector<RandomPoint3d> _pts){
	    pts  = _pts;
	}
};


class FrameLine
{
public:
    cv::Point2d 	p,q;    // image endpoints p and q (point2d)
    cv::Mat     	l;  
    double 			lineEq2d[3];//3-vector equation

    bool 			haveDepth;
    RandomLine3d 	line3d;

    cv::Point2d 	r;  //gradient direction
    cv::Mat 		des; //descriptor

    int 			lid;  //local id in frame
    int 			gid;  //global id 
    int 			lid_prvKfrm;  //correspondence's lid in previous keyframe

    FrameLine(){gid=-1;}
    FrameLine(cv::Point2d _p, cv::Point2d _q);
    ~FrameLine(){}
    cv::Point2d getGradient(cv::Mat* xGradient, cv::Mat* yGradient);
    void compLineEq2d()
    {
        cv::Mat pt1 = (cv::Mat_<double>(3,1)<<p.x, p.y, 1);
        cv::Mat pt2 = (cv::Mat_<double>(3,1)<<q.x, q.y, 1);
        cv::Mat lnEq = pt1.cross(pt2);
        lnEq = lnEq/sqrt(lnEq.at<double>(0)*lnEq.at<double>(0)
                +lnEq.at<double>(1)*lnEq.at<double>(1)); //normalize
        lineEq2d[0]=lnEq.at<double>(0);
        lineEq2d[1]=lnEq.at<double>(1);
        lineEq2d[2]=lnEq.at<double>(2);
    }
};

class Frame
{
public:
    long unsigned int         	id;
    static long unsigned int	nextid;
    double      				timestamp;
    bool        				isKeyFrame;
    vector<FrameLine> 			lines;
    cv::Mat     				R, t;
    cv::Mat     				rgb, gray;
    cv::Mat     				depth, oriDepth;
	string 						rgbname;
    
    //camera
    static bool					mbInitialFlag;
    static Camera 				camera;  //fx,fy,cx,cy,scale,k1,k2,p1,p2,k3
    static cv::Mat 				K;
    static cv::Mat 				distCoeffs;
	
	//image boundary
	static float 				mnMinX;
	static float				mnMaxX;
	static float				mnMinY;
	static float				mnMaxY;
    
    //line
    double     	 				lineLenThresh;
    
    //orb
    static ORBextractor 		*orbextractor;
    cv::Mat 					mDescriptors;
    vector<cv::KeyPoint> 		mvKeypoints;
    vector<cv::KeyPoint>     	mvKeypointsUn;  //UnDistortation
    int  						N;  //number of keypoints
    
    vector<cv::KeyPoint>		feature_locations_2d_;
    vector<Eigen::Vector4f>		feature_locations_3d_;
    
    //BoW
    //ORBVocabulary	  			*mpORBVocabulary;
    //DBoW2::BowVector  			mBowVec;
    //DBoW2::FeatureVector		mFeatureVec;
    
    //POSE
    cv::Mat 					mTcw;    //Camera pose
    cv::Mat 					mRcw;    //Rotation
    cv::Mat  					mtcw;    //translation
    cv::Mat 					mRwc;    //Rotation inverse
    cv::Mat    					mOw;     //=mtwc  //camera center
    

    Frame(){}
    ~Frame(){}
    Frame(double _timestamp, string  rgb_filename, string depth_filename, Camera _camera);
    Frame(double _timestamp, cv::Mat _rgb, cv::Mat _depth, Camera _camera);
    
    void undistortKeypoints();
	void computeImageBoundary();
    
   
    //line feature
    void detectFrameLines(int method = 0);
    void extractLineDepth();
    void clear();
    void write2file(string filename);
    
    //orb
    void extractORB();
    void projectKeypointTo3d();
    void computeBow();
    void setPose(cv::Mat Tcw);
    
    //Point cloud
    PointCloud::Ptr img2cloud();
	
#ifdef SEGMENT
	//groud segmentation
	bool segOK;
	static PyObject* pNet;
	static PyObject* pFunc;
	
	vector<double> c;
	
	void segmentInit();
	void segment();
#endif
	void getEquation(PointCloud::Ptr cloud, vector<double>& c);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr planeExtraction();
	void planeEquation(vector<vector<double>>& equations);
	
	//fast plane extraction 
	void AHCPlane();
	
};

/*
class PoseConstraint
{
public:
	int 	from, to; // keyframe ids
	cv::Mat R, t;
	int 	numMatches;
};*/


class SystemParameters
{
public:
	double	pt2line_dist_extractline;	// threshold pt to line distance when detect lines from pts
	double	pt2line_mahdist_extractline;// threshold for pt to line mahalanobis distance
	double	line_segment_len_thresh;	// min lenght of image line segment to use 
	double	ratio_support_pts_on_line;	// the ratio of the number of filled cells over total cell number along a line
										// to check if a line has enough points covering the whole range
	int		num_cells_lineseg_range;	// divide a linesegment into multiple cells 

	int 	line_sample_max_num;
	int 	line_sample_min_num;
	double 	line_sample_interval;
	int 	line3d_mle_iter_num;
	int 	line_detect_algorithm;
	double 	msld_sample_interval;
	int 	ransac_iters_line_motion;
	int 	adjacent_linematch_window;
	int 	line_match_number_weight;
	int 	min_feature_matches;
	double 	max_mah_dist_for_inliers;
	double  g2o_line_error_weight;
	int 	min_matches_loopclose;
	
	int		num_2dlinematch_keyframe;	// detect keyframe, minmum number of 2d line matches left
	int		num_3dlinematch_keyframe;
	double	pt2line3d_dist_relmotion;	// in meter, 
	double  line3d_angle_relmotion;		// in degree
	int		num_raw_frame_skip;			// number of raw frame to skip when tracking lines
	int		window_length_keyframe;		
	bool	fast_motion;
	double	inlier_ratio_constvel;
	int		num_pos_lba;
	int		num_frm_lba;
	// ----- lsd setting -----
	double 	lsd_angle_th;
	double 	lsd_density_th;
	// ----- loop closing -----
	double 	loopclose_interval;  // frames, check loop closure
	int		loopclose_min_3dmatch;  // min_num for 3d line matches between two frames
	
	bool 	dark_lighting;
	double	max_img_brightness;

	SystemParameters()
	{
	    // ----- 2d-line -----
	    line_segment_len_thresh		= 10;// pixels, min lenght of image line segment to use 
	    msld_sample_interval		= 1;
		
	    // ----- 3d-line measurement ----
	    line_sample_max_num			= 100;
	    line_sample_min_num			= 10;
	    line_sample_interval		= 1;
	    line3d_mle_iter_num			= 50; 
	    
	    //extract3dLine
	    pt2line_dist_extractline	= 0.02;	// meter, threshold pt to line distance when detect lines from pts
	    pt2line_mahdist_extractline	= 1.5;	//  NA,	  as above
	    
	    //verify3dLine
	    num_cells_lineseg_range		= 10;	// 1, 
	    ratio_support_pts_on_line	= 0.5;	// ratio, when verifying a detected 3d line
	    

	    // ----- key frame -----
	    num_raw_frame_skip			= 1;
	    window_length_keyframe		= 10;
	    num_2dlinematch_keyframe	= 5; 	// detect keyframe, minmum number of 2d line matches left
	    num_3dlinematch_keyframe	= 3;
    
	    // ----- relative motion -----
	    pt2line3d_dist_relmotion	= 0.05;	// in meter, 
	    line3d_angle_relmotion		= 10;
	    fast_motion					= 1;
	    inlier_ratio_constvel		= 0.4;
	    dark_lighting				= false;
	    max_img_brightness			= 0;
	    ransac_iters_line_motion	= 50;
	    adjacent_linematch_window 	= 10;
	    
	    line_match_number_weight    = 1; //0.5
	    min_feature_matches 		= 3;
	    max_mah_dist_for_inliers 	= 3;
	    g2o_line_error_weight 		= 1.0;
	    min_matches_loopclose 		= 20;
	    
	    // ----- lba -----
	    num_pos_lba					= 5;
	    num_frm_lba					= 7;
    
	    // ----- loop closing -----
	    loopclose_interval			= 50;  // frames, check loop closure
	    loopclose_min_3dmatch		= 30;  // min_num for 3d line matches between two frames
    
	    // ----- lsd setting -----
	    lsd_angle_th 				= 40;   //  22.5
	    lsd_density_th				= 0.7;
	}
	
};



#endif
