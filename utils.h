#ifndef LINESLAM_UTILS_H
#define LINESLAM_UTILS_H

#include "base.h"
#include "lsd.h"
#include "frame.h"
#include "edge_se3_lineendpts.h"
#include "vertex_lineendpts.h"

struct LS {
  double sx, sy, ex, ey; // Start & end coordinates of the line segment
};

template<class bidiiter> //Fisher-Yates shuffle
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
	size_t left = std::distance(begin, end);
	while (num_random--) {
		bidiiter r = begin;
		std::advance(r, rand()%left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}    
	return begin;
}

void setPCLBackGround(pcl::visualization::PCLVisualizer& viewer);

cv::Mat array2mat(double a[], int n);
cv::Mat cvpt2mat(cv::Point2d p, bool homo = true);
cv::Mat cvpt2mat(cv::Point3d p, bool homo = true);
cv::Point2d mat2cvpt2d(cv::Mat m);
cv::Point3d mat2cvpt3d(cv::Mat m);
string num2str(double i);
cv::Mat vec2SkewMat(cv::Mat vec);
cv::Mat vec2SkewMat (cv::Point3d vec);
cv::Mat toCvMat(g2o::SE3Quat SE3);
cv::Mat toCvMat(Eigen::Matrix<double,4,4> m);

cv::Mat q2r(cv::Mat q);
cv::Mat q2r (double* q);
cv::Mat r2q(cv::Mat R);
Eigen::Vector4d r2q(Eigen::Matrix3d R);

//match
vector<cv::DMatch> GmsMatch(Frame& frame1,Frame& frame2, vector<cv::DMatch> matches);
vector<cv::DMatch> FilterMatch(Frame& frame1,Frame& frame2, vector<cv::DMatch> matches);

//line detector and MSLD descriptor
ntuple_list callLsd(IplImage* src);
LS *DetectLinesByED(unsigned char *srcImg, int width, int height, int *pNoLines);
LS* callEDLines (const cv::Mat& im_uchar, int* numLines);
int computeSubPSR(cv::Mat* xGradient, cv::Mat* yGradient, cv::Point2d p, double s, cv::Point2d g, vector<double>& vs);
int computeMSLD(FrameLine& l, cv::Mat* xGradient, cv::Mat* yGradient);


/*************************Extract Line***************************/
/****************************************************************/
double dist3d_pt_line (cv::Point3d X, cv::Point3d A, cv::Point3d B);
double mah_dist3d_pt_line (const RandomPoint3d& pt, const cv::Point3d& q1, const cv::Point3d& q2);

void computeLine3d_svd (vector<cv::Point3d> pts, cv::Point3d& mean, cv::Point3d& drct);
void computeLine3d_svd (const vector<RandomPoint3d>& pts, const vector<int>& idx, cv::Point3d& mean, cv::Point3d& drct);

RandomLine3d extract3dline(const vector<cv::Point3d>& pts,SystemParameters sysPara);
RandomLine3d extract3dline_mahdist(const vector<RandomPoint3d>& pts,SystemParameters sysPara);

cv::Point3d projectPt3d2Ln3d (const cv::Point3d& P, const cv::Point3d& mid, const cv::Point3d& drct);
cv::Point3d projectPt3d2Ln3d_2 (const cv::Point3d& P, const cv::Point3d& A, const cv::Point3d& B);
bool verify3dLine(vector<cv::Point3d> pts, cv::Point3d A, cv::Point3d B,SystemParameters sysPara);
bool verify3dLine(const vector<RandomPoint3d>& pts, const cv::Point3d& A,  const cv::Point3d& B,SystemParameters sysPara);


double depthStdDev (double d);
RandomPoint3d compPt3dCov (cv::Point3d pt, cv::Mat K);
Eigen::Matrix3f compPt3dCov (Eigen::Vector3f pt, double fx,double fy, double cu, double cv);

cv::Point3d mahvec_3d_pt_line(const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
cv::Point3d mahvec_3d_pt_line(const RandomPoint3d& pt, cv::Mat q1, cv::Mat q2);
cv::Point3d closest_3dpt_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
double closest_3dpt_ratio_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);

void MLEstimateLine3d (RandomLine3d& line,int maxIter);
cv::Mat MlELine3dCov(const vector<RandomPoint3d>& pts, int idx1, int idx2, const double l[6]);
cv::Mat jac_rpt2ln_mahvec_wrt_ln(const RandomPoint3d& pt, const double l[6]) ;
cv::Mat jac_pt2pt_mahvec_wrt_pt (const RandomPoint3d& pt, const double A[3]) ;


/*************************Match Line*****************************/
/****************************************************************/
double pt_to_line_dist2d(const cv::Point2d& p, const double l[3]);
double line_to_line_dist2d(FrameLine& a, FrameLine& b);
double projectPt2d_to_line2d(const cv::Point2d& X, const cv::Point2d& A,const cv::Point2d& B);
double lineSegmentOverlap( FrameLine& a,  FrameLine& b);
void matchLine(vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int>>& matches);
void trackLine (vector<FrameLine>& f1, vector<FrameLine>& f2, vector<vector<int> >& matches,SystemParameters sysPara);


double ave_img_bright(cv::Mat img);
double pesudoHuber(double e, double band);
//void write2file (Map3d& m, string suffix);
void write_linepairs_tofile(vector<RandomLine3d> a, vector<RandomLine3d> b, string fname, double timestamp);


/*************************Motion*********************************/
/****************************************************************/
Eigen::Matrix4f getTransformFromMatches(const Frame* newer_node, const Frame* earlier_node,
           const std::vector<cv::DMatch>& matches,
           bool& valid, const float max_dist_m);

//!Do sparse bundle adjustment for the node pair
void getTransformFromMatchesG2O(const Frame* earlier_node,const Frame* newer_node,
           const std::vector<cv::DMatch> & matches,
           Eigen::Matrix4f& transformation_estimate, //Input (initial guess) and Output
           int iterations );

Eigen::Isometry3d getTransformFromHybridMatchesG2O (const Frame* earlier_node,const Frame* newer_node,
           const std::vector<cv::DMatch> & pt_matches,
           const std::vector<cv::DMatch> & ln_matches,
           Eigen::Matrix4f& transformation_estimate, //Input (initial guess) and Output
           int iterations, SystemParameters sysPara);

void costFun_MLEstimateLine3d(double *p, double *error, int m, int n, void *adata);
bool computeRelativeMotion_svd (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);

Eigen::Matrix4f getTransform_Lns_Pts_pcl(const Frame* trainNode, const Frame* queryNode, 
		const std::vector<cv::DMatch>& point_matches,
		const std::vector<cv::DMatch>& line_matches,bool& valid);

Eigen::Isometry3d getTransform_Line_svd(const Frame* trainNode, const Frame* queryNode, 
		const std::vector<cv::DMatch>& matches,bool& valid);
		
Eigen::Isometry3d getTransform_PtsLines_ransac (const Frame* trainNode, const Frame* queryNode, 
		const std::vector<cv::DMatch> all_point_matches,const std::vector<cv::DMatch> all_line_matches,
		std::vector<cv::DMatch>& output_point_inlier_matches, std::vector<cv::DMatch>& output_line_inlier_matches,
		Eigen::Matrix4f& ransac_tf, float& inlier_rmse, SystemParameters sysPara);

Eigen::Matrix4f getIcpAlignment(Frame* queryNode, Frame* trainNode);

#endif
