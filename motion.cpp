#include "utils.h"
#include "frame.h"
#include "edge_se3_lineendpts.h"

#define OPT_USE_MAHDIST
#define MOTION_USE_MAHDIST

#define POINT
#define LINE
#define VERBOSE


inline double depth_std_dev(double depth)
{
  static double depth_std_dev = 0.006;
  // Previously used 0.006 
  //http://www.ros.org/wiki/openni_kinect/kinect_accuracy;
  return depth_std_dev * depth * depth;
}

//Functions without dependencies
inline double depth_covariance(double depth)
{
  double stddev = depth_std_dev(depth);
  return stddev * stddev;
}

inline Eigen::Matrix3d point_information_matrix(double distance)
{
  Eigen::Matrix3d inf_mat = Eigen::Matrix3d::Identity();
  inf_mat(2,2) = 1.0/depth_covariance(distance);
  return inf_mat;
}


//!Set parameters for icp optimizer
void optimizerSetup(g2o::SparseOptimizer& optimizer,cv::Mat K)
{
    //sparse optimizer
    g2o::BlockSolverX::LinearSolverType * linearSolver = new g2o::LinearSolverCholmod<g2o ::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * blocksolver = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(blocksolver);
    optimizer.setAlgorithm(algorithm);
    
    //camera parameters
    g2o::ParameterCamera* cameraParams = new g2o::ParameterCamera();
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);
    cameraParams->setKcam(fx,fy,cx,cy);
    g2o::SE3Quat offset; // identity
    cameraParams->setOffset(offset);
    
    cameraParams->setId(0);
    optimizer.addParameter(cameraParams);
}


//first cam is fixed, transformation from the first to the second cam will be computed
std::pair<g2o::VertexSE3*, g2o::VertexSE3*>  sensorVerticesSetup(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& tf_estimate)
{
    g2o::VertexSE3 *vc1 = new VertexSE3();
    Eigen::Quaterniond q(tf_estimate.topLeftCorner<3,3>().cast<double>());//initialize rotation from estimate 
    Eigen::Vector3d    t(tf_estimate.topRightCorner<3,1>().cast<double>());//initialize translation from estimate
    g2o::SE3Quat cam(q,t);
    vc1->setEstimate(g2o::SE3Quat());
    vc1->setId(0);  
    vc1->setFixed(true);
    optimizer.addVertex(vc1);
  
    g2o::VertexSE3 *vc2 = new VertexSE3();
    vc2->setEstimate(cam);
    vc2->setId(1); 
    vc2->setFixed(false);
    optimizer.addVertex(vc2);
    
    // add to optimizer
    return std::make_pair(vc1, vc2);
}

g2o::EdgeSE3PointXYZDepth* edgeToFeature(const Frame* node, unsigned int feature_id,
                              g2o::VertexSE3* camera_vertex, g2o::VertexPointXYZ* feature_vertex)
{
    g2o::EdgeSE3PointXYZDepth* edge = new g2o::EdgeSE3PointXYZDepth();
    cv::KeyPoint kp = node->feature_locations_2d_[feature_id];
    Vector4f position = node->feature_locations_3d_[feature_id];
    float d = position(2);
  
    feature_vertex->setEstimate(position.cast<double>().head<3>());
    feature_vertex->setFixed(false);
    
    //cout<<kp.pt.x<<" "<<kp.pt.y<<" "<<d<<endl;
    Eigen::Vector3d pix_d(kp.pt.x,kp.pt.y,d);  //
    edge->setMeasurement(pix_d);
    Eigen::Matrix3d info_mat = point_information_matrix(d);
    edge->setInformation(info_mat);
    
 
    edge->setParameterId(0,0);
    edge->setRobustKernel(new g2o::RobustKernelHuber());  //true
    edge->vertices()[0] = camera_vertex;
    edge->vertices()[1] = feature_vertex;
   
   
    return edge;
}

//!Compute 
void getTransformFromMatchesG2O(const Frame* earlier_node, const Frame* newer_node,
                                const std::vector<cv::DMatch> & matches,
                                Eigen::Matrix4f& transformation_estimate, 
                                int iterations)
{
    std::vector<cv::DMatch> matches_with_depth;
    BOOST_FOREACH(const cv::DMatch& m, matches)
    {
      
		Point2f kp1 = earlier_node->feature_locations_2d_[m.queryIdx].pt;
		Point2f kp2 = newer_node->feature_locations_2d_[m.trainIdx].pt;

		
		float depth1 = earlier_node->feature_locations_3d_[m.queryIdx](2);
		float depth2 = newer_node->feature_locations_3d_[m.trainIdx](2);
		
		if( depth1>=1e-2&&depth1<=10&&depth2>=1e-2&&depth2<=10
		&&(kp1.x>=1&&kp1.x<=640&&kp1.y>=1&&kp1.y<=480)
		&&(kp2.x>=1&&kp2.x<=640&&kp2.y>=1&&kp2.y<=480))
		{
			matches_with_depth.push_back(m);
		}
    }
    
  
    cout<<"matches_all       :"<<matches.size()<<endl;
    cout<<"matches_with_depth:"<<matches_with_depth.size()<<endl;

    g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();
    optimizerSetup(*optimizer, Frame::K);
    Eigen::Matrix4f tfinv = transformation_estimate;//Eigen::Matrix4f::Identity();
    std::pair<g2o::VertexSE3*, g2o::VertexSE3*> cams = sensorVerticesSetup(*optimizer, tfinv);

    int v_id = optimizer->vertices().size(); //0 and 1 are taken by sensor vertices
    
    //For each match, create a vertex and connect it to the sensor vertices with the measured position
    BOOST_FOREACH(const cv::DMatch& m, matches_with_depth)
    {
		g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ();
		v->setId(v_id++);
		v->setMarginalized(true);
		v->setFixed(false);
		optimizer->addVertex(v);
		
		g2o::EdgeSE3PointXYZDepth* e1 = edgeToFeature(earlier_node, m.queryIdx, cams.first,  v);
		optimizer->addEdge(e1);
		
		g2o::EdgeSE3PointXYZDepth* e2 = edgeToFeature(newer_node,   m.trainIdx, cams.second, v);
		optimizer->addEdge(e2);  
    }
    
    //start optimization
    optimizer->setVerbose(false);
    optimizer->initializeOptimization();
    std::cout<<"g2o error "<< optimizer->activeChi2();
    optimizer->optimize(100);
    std::cout<<" => " <<optimizer->activeChi2()<<std::endl;
    
    transformation_estimate = cams.second->estimate().cast<float>().matrix();
    delete optimizer;
}


Eigen::Isometry3d getTransformFromHybridMatchesG2O (const Frame* earlier_node, const Frame* newer_node,
                                       const std::vector<cv::DMatch> & pt_matches,
                                       const std::vector<cv::DMatch> & ln_matches,
                                       Eigen::Matrix4f& transformation_estimate, 
                                       int iterations,  SystemParameters sysPara)
{
  
	std::vector<cv::DMatch> matches_with_depth;
	BOOST_FOREACH(const cv::DMatch& m, pt_matches)
	{

		Point2f kp1 = earlier_node->feature_locations_2d_[m.queryIdx].pt;
		Point2f kp2 = newer_node->feature_locations_2d_[m.trainIdx].pt;

		
		float depth1 = earlier_node->feature_locations_3d_[m.queryIdx](2);
		float depth2 = newer_node->feature_locations_3d_[m.trainIdx](2);
		
		if( depth1>=1e-2&&depth1<=10&&depth2>=1e-2&&depth2<=10
			&&(kp1.x>=1&&kp1.x<=640&&kp1.y>=1&&kp1.y<=480)
			&&(kp2.x>=1&&kp2.x<=640&&kp2.y>=1&&kp2.y<=480))
		{
			matches_with_depth.push_back(m);
		}
	}
 
	//G2O Initialization
	g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();
	optimizerSetup(*optimizer, Frame::K);
	
	Eigen::Matrix4f tfinv = transformation_estimate;//Eigen::Matrix4f::Identity();//transformation_estimate.inverse();
	std::pair<g2o::VertexSE3*, g2o::VertexSE3*> cams = sensorVerticesSetup(*optimizer, tfinv);
	int v_id = optimizer->vertices().size(); //0 and 1 are taken by sensor vertices
	
	vector<g2o::EdgeSE3PointXYZ*>   pt_edges;
	vector<g2o::EdgeSE3LineEndpts*> ln_edges;

	//add the parameter representing the sensor offset  
	g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(Eigen::Isometry3d::Identity());
	sensorOffset->setId(1);
	optimizer->addParameter(sensorOffset);

	//camera intrinsic parameters 
	double  fx = Frame::K.at<double>(0,0), 
			fy = Frame::K.at<double>(1,1),
			cu = Frame::K.at<double>(0,2),
			cv = Frame::K.at<double>(1,2);

#ifdef SEGMENT
	{
		for(int ii=0; ii<1; ii++)
		{
			Eigen::Vector3d norm_older(earlier_node->c[0],earlier_node->c[1],earlier_node->c[2]);
			Eigen::Vector3d norm_newer(newer_node->c[0],newer_node->c[1],newer_node->c[2]);
			
			g2o::VertexPointXYZ* v= new g2o::VertexPointXYZ();
			v->setEstimate(norm_newer);
			v->setId(v_id++);
			v->setMarginalized(true);
			v->setFixed(false);
			optimizer->addVertex(v);

			g2o::EdgeSE3PointXYZ* e_older = new g2o::EdgeSE3PointXYZ();
			e_older->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.first); 
			e_older->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
			e_older->setMeasurement(norm_older);
			//e_older->information() = compPt3dCov(pt_older, fx, fy, cu, cv).cast<double>().inverse();
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			rk->setDelta(5.99);
			e_older->setRobustKernel(rk);
			e_older->setParameterId(0,1);
			optimizer->addEdge(e_older);
			
			g2o::EdgeSE3PointXYZ* e_newer = new g2o::EdgeSE3PointXYZ();
			e_newer->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.second); 
			e_newer->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
			e_newer->setMeasurement(norm_newer);
			//e_newer->information() = compPt3dCov(pt_newer, fx, fy, cu, cv).cast<double>().inverse();
			g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
			rk1->setDelta(5.99);
			e_newer->setRobustKernel(rk1);
			e_newer->setParameterId(0,1);
			optimizer->addEdge(e_newer);
			
			pt_edges.push_back(e_older);
			pt_edges.push_back(e_newer);
		}
	}	
#endif
		
		
	  
      // ************** point matches *****************//
#ifdef POINT
	//cout<<"Point matches:"<<matches_with_depth.size();
	BOOST_FOREACH(const cv::DMatch& m, matches_with_depth)
	{
		Eigen::Vector3f pt_older = earlier_node->feature_locations_3d_[m.queryIdx].head<3>();
		Eigen::Vector3f pt_newer = newer_node->feature_locations_3d_[m.trainIdx].head<3>();

		g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ(); 
		v->setEstimate(pt_newer.cast<double>());
		v->setId(v_id++);
		v->setMarginalized(true);
		v->setFixed(false);
		optimizer->addVertex(v);
		
		g2o::EdgeSE3PointXYZ* e_older = new g2o::EdgeSE3PointXYZ();
		e_older->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.first); 
		e_older->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
		e_older->setMeasurement(pt_older.cast<double>());
		e_older->information() = compPt3dCov(pt_older, fx, fy, cu, cv).cast<double>().inverse();
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		rk->setDelta(5.99);
		e_older->setRobustKernel(rk);
		e_older->setParameterId(0,1);
		optimizer->addEdge(e_older);
		
		g2o::EdgeSE3PointXYZ* e_newer = new g2o::EdgeSE3PointXYZ();
		e_newer->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.second); 
		e_newer->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
		e_newer->setMeasurement(pt_newer.cast<double>());
		e_newer->information() = compPt3dCov(pt_newer, fx, fy, cu, cv).cast<double>().inverse();
		g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
		rk1->setDelta(5.99);
		e_newer->setRobustKernel(rk1);
		e_newer->setParameterId(0,1);
		optimizer->addEdge(e_newer);
		
		pt_edges.push_back(e_older);
		pt_edges.push_back(e_newer);
		
	}
#endif
  
      //*************** line matches ****************//
#ifdef LINE
      //cout<<"    Line matches:"<<ln_matches.size()<<endl;
      BOOST_FOREACH(const cv::DMatch& m, ln_matches) 
      {
		g2o::VertexLineEndpts* v = new g2o::VertexLineEndpts();
		Eigen::Vector6d line_new;
		
		line_new << newer_node->lines[m.trainIdx].line3d.A.x, newer_node->lines[m.trainIdx].line3d.A.y, newer_node->lines[m.trainIdx].line3d.A.z,
				newer_node->lines[m.trainIdx].line3d.B.x, newer_node->lines[m.trainIdx].line3d.B.y, newer_node->lines[m.trainIdx].line3d.B.z; 
		
		//cout<<"Line new:"<<line_new<<endl;
		// line represented wrt the newer node (second, fixed one)
		v->setEstimate(line_new); 
		v->setId(v_id++);
		v->setMarginalized(true);
		v->setFixed(false);
		optimizer->addVertex(v);

		// create edges between line and cams    
		// newer node
		g2o::EdgeSE3LineEndpts * e_newer = new g2o::EdgeSE3LineEndpts();
		e_newer->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.second); 
		e_newer->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
		e_newer->setMeasurement(line_new);
		e_newer->information() = Matrix6d::Identity() ;//* sysPara.g2o_line_error_weight;  // must be identity!
		cv::Mat covA = newer_node->lines[m.trainIdx].line3d.rndA.cov;
		cv::Mat covB = newer_node->lines[m.trainIdx].line3d.rndB.cov;            
		e_newer->endptCov = Matrix6d::Identity();
		for(int ii=0; ii<3; ++ii) {
			for(int jj=0; jj<3; ++jj) {
			e_newer->endptCov(ii,jj) = covA.at<double>(ii,jj);
			e_newer->endptCov(ii+3,jj+3) = covB.at<double>(ii,jj);
			}
		}       

		{
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_newer->endptCov.block<3,3>(0,0),Eigen::ComputeFullU);
			Eigen::Matrix3d D_invsqrt;
			D_invsqrt.fill(0);
			D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
			D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
			D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
			Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
			e_newer->endpt_AffnMat.block<3,3>(0,0) = am;
		}
		{
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_newer->endptCov.block<3,3>(3,3),Eigen::ComputeFullU);
			Eigen::Matrix3d D_invsqrt;
			D_invsqrt.fill(0);
			D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
			D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
			D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
			Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
			e_newer->endpt_AffnMat.block<3,3>(3,3) = am;
		}
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		rk->setDelta(5.99);
		e_newer->setRobustKernel(rk);
		e_newer->setParameterId(0,1);// param id 0 of the edge corresponds to param id 1 of the optimizer     
		optimizer->addEdge(e_newer);    
		
		//// edge to older node
		g2o::EdgeSE3LineEndpts* e_older = new g2o::EdgeSE3LineEndpts();
		e_older->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.first); 
		e_older->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
		Eigen::Vector6d line_older;
		line_older << earlier_node->lines[m.queryIdx].line3d.A.x, earlier_node->lines[m.queryIdx].line3d.A.y, earlier_node->lines[m.queryIdx].line3d.A.z,
				earlier_node->lines[m.queryIdx].line3d.B.x, earlier_node->lines[m.queryIdx].line3d.B.y, earlier_node->lines[m.queryIdx].line3d.B.z;
		e_older->setMeasurement(line_older);
		e_older->information() = Eigen::Matrix6d::Identity();// * sysPara.g2o_line_error_weight;  // must be identity!
		covA = earlier_node->lines[m.queryIdx].line3d.rndA.cov;            
		covB = earlier_node->lines[m.queryIdx].line3d.rndB.cov;
		e_older->endptCov = Eigen::Matrix6d::Identity() ;
		for(int ii=0; ii<3; ++ii) {
			for(int jj=0; jj<3; ++jj) {
			e_older->endptCov(ii,jj) = covA.at<double>(ii,jj);
			e_older->endptCov(ii+3,jj+3) = covB.at<double>(ii,jj);
			}
		}

		{
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_older->endptCov.block<3,3>(0,0),Eigen::ComputeFullU);
			Eigen::Matrix3d D_invsqrt;
			D_invsqrt.fill(0);
			D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
			D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
			D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
			Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
			e_older->endpt_AffnMat.block<3,3>(0,0) = am;
		}
		
		{
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_older->endptCov.block<3,3>(3,3),Eigen::ComputeFullU);
			Eigen::Matrix3d D_invsqrt;
			D_invsqrt.fill(0);
			D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
			D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
			D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
			Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
			e_older->endpt_AffnMat.block<3,3>(3,3) = am;
		}
		g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
		rk1->setDelta(5.99);
		e_older->setRobustKernel(rk1);
		e_older->setParameterId(0,1);
		optimizer->addEdge(e_older); 

		ln_edges.push_back(e_newer);
		ln_edges.push_back(e_older); 
      }
#endif

	optimizer->setVerbose(false);
	optimizer->initializeOptimization();
#ifdef VERBOSE
	double pterr = 0, lnerr = 0;
	for(int i=0; i<pt_edges.size();++i)
	{
		pt_edges[i]->computeError();
		pterr += pt_edges[i]->chi2();
	}
	for(int i=0; i<ln_edges.size();++i) 
	{
		ln_edges[i]->computeError();
		lnerr += ln_edges[i]->chi2();
	}
	//cout<<"Original:"<<pterr<<" \t "<<lnerr<<" \t "<<lnerr/(pterr+lnerr)<<endl;
#endif
      
      optimizer->optimize(100);
      
#ifdef VERBOSE
      pterr = 0, lnerr = 0;
      for(int i=0; i<pt_edges.size();++i) {
		pt_edges[i]->computeError();
		pterr += pt_edges[i]->chi2();
      }
      for(int i=0; i<ln_edges.size();++i) {
		ln_edges[i]->computeError();
		lnerr += ln_edges[i]->chi2();
      }
      //cout<<"After:   "<<pterr<<" \t "<<lnerr<<" \t "<<lnerr/(pterr+lnerr)<<endl;
#endif
      /*
       *Eigen::Matrix4f rst = vc1->estimate().cast<float>().inverse().matrix();
		for(int i=0; i<3; ++i) {
			t.at<double>(i) = rst(i,3);
			for(int j=0; j<3; ++j) 
				R.at<double>(i,j) = rst(i,j);
		}
       */
      transformation_estimate = cams.second->estimate().cast<float>().matrix();
      delete optimizer;
	  return cams.second->estimate();
}  



bool computeRelativeMotion_svd (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t)
	// input needs at least 2 correspondences of non-parallel lines
	// the resulting R and t works as below: x'=Rx+t for point pair(x,x');
{
	if(a.size()<2)	{return false;}
	// convert to the representation of Zhang's paper
	for(int i=0; i<a.size(); ++i) {
		cv::Point3d l, m;
		if(cv::norm(a[i].u)<0.9) {
			l = a[i].B - a[i].A;
			m = (a[i].A + a[i].B) * 0.5;
			a[i].u = l * (1/cv::norm(l));
			a[i].d = a[i].u.cross(m);
		}
		if(cv::norm(b[i].u)<0.9){		
			l = b[i].B - b[i].A;
			m = (b[i].A + b[i].B) * 0.5;
			b[i].u = l * (1/cv::norm(l));
			b[i].d = b[i].u.cross(m);
		}
	}

	cv::Mat A = cv::Mat::zeros(4,4,CV_64F);
	for(int i=0; i<a.size(); ++i) {
		cv::Mat Ai = cv::Mat::zeros(4,4,CV_64F);
		Ai.at<double>(0,1) = (a[i].u-b[i].u).x;
		Ai.at<double>(0,2) = (a[i].u-b[i].u).y;
		Ai.at<double>(0,3) = (a[i].u-b[i].u).z;
		Ai.at<double>(1,0) = (b[i].u-a[i].u).x;
		Ai.at<double>(2,0) = (b[i].u-a[i].u).y;
		Ai.at<double>(3,0) = (b[i].u-a[i].u).z;
		vec2SkewMat(a[i].u+b[i].u).copyTo(Ai.rowRange(1,4).colRange(1,4));
		A = A + Ai.t()*Ai;
	}
	cv::SVD svd(A);
	cv::Mat q = svd.u.col(3);
	//cout<<"q="<<q<<endl;
	R = q2r(q);
	cv::Mat uu = cv::Mat::zeros(3,3,CV_64F),
		udr= cv::Mat::zeros(3,1,CV_64F);
	for(int i=0; i<a.size(); ++i) {
		uu = uu + vec2SkewMat(b[i].u)*vec2SkewMat(b[i].u).t();
		udr = udr + vec2SkewMat(b[i].u).t()* (cvpt2mat(b[i].d,0)-R*cvpt2mat(a[i].d,0));
	}
	t = uu.inv()*udr;	
	return true;
}

bool isSamePlane(vector<double> a, vector<double> b)
{
	if(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]>0.95&&abs(a[3]-b[3])<0.05)
	{
		cout<<"A:"<<a[0]<<" "<<a[1]<<" "<<a[2]<<" "<<a[3]<<endl;
		cout<<"B:"<<b[0]<<" "<<b[1]<<" "<<b[2]<<" "<<b[3]<<endl;
		return true;
	}
	else return false;
}

Eigen::Matrix4f getIcpAlignment(Frame* queryNode,  Frame* trainNode)
{
	PointCloud::Ptr cloud1(new PointCloud);
	PointCloud::Ptr cloud2(new PointCloud); 
	
	//cloud1 = queryNode->img2cloud();
	//cloud2 = trainNode->img2cloud();
	
	cloud1 = queryNode->planeExtraction();
	cloud2 = trainNode->planeExtraction();
	
	
    ///////////////////////////
	/*
	MyTimer timer;
	timer.start();
	vector<vector<double>> e1,e2;
	queryNode->planeEquation(e1);
	trainNode->planeEquation(e2);
	
	timer.end();
	timer.start();
	
	cout<<"Plane matches:"<<endl;
	for(int i=0; i<e1.size();i++)
	{
		for(int j=0; j<e2.size();j++)
		{
			isSamePlane(e1[i],e2[j]);
		}
	}
	timer.end();
	*/
	//////////////////////////////
	
	//pcl::visualization::CloudViewer viewer ("Cluster viewer");
	//viewer.showCloud(cloud2);
	//while (!viewer.wasStopped ()){}
	
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setInputCloud(cloud1);
	icp.setInputTarget(cloud2);
	PointCloud Final;
	icp.align(Final);
	cout << "has converged:" << icp.hasConverged() 
		<<" score: " <<icp.getFitnessScore() << endl;
	//cout << icp.getFinalTransformation() <<endl;
	
	return icp.getFinalTransformation();
}


Eigen::Matrix4f getTransform_Lns_Pts_pcl( const Frame* queryNode, const Frame* trainNode,
					const std::vector<cv::DMatch>& point_matches_ori,
					const std::vector<cv::DMatch>& line_matches, bool& valid)
// Note: the result transforms a point from the queryNode's CS to the trainNode's CS
{
    std::vector<cv::DMatch> point_matches;
	BOOST_FOREACH(const cv::DMatch& m, point_matches_ori)
	{
	    Point2f kp1 = queryNode->feature_locations_2d_[m.queryIdx].pt;
	    Point2f kp2 = trainNode->feature_locations_2d_[m.trainIdx].pt;

	    
	    float depth1 = queryNode->feature_locations_3d_[m.queryIdx](2);
	    float depth2 = trainNode->feature_locations_3d_[m.trainIdx](2);
	    
	    if( depth1>=1e-3&&depth1<=10&&depth2>=1e-3&&depth2<=10
	      &&(kp1.x>=1&&kp1.x<=640&&kp1.y>=1&&kp1.y<=480)
	      &&(kp2.x>=1&&kp2.x<=640&&kp2.y>=1&&kp2.y<=480))
	    {
			point_matches.push_back(m);
	    }
	}
    
	if(point_matches.size()<1 || point_matches.size()+line_matches.size()<3) { 
		valid = false;
		return Eigen::Matrix4f();
	}
	pcl::TransformationFromCorrespondences tfc;

	//// project the point to all lines
	for(int i=0; i < line_matches.size(); ++i) 
	{
		int ptidx = rand()%point_matches.size();
		cv::Point3d query_pt(queryNode->feature_locations_3d_[point_matches[ptidx].queryIdx](0),
  				     queryNode->feature_locations_3d_[point_matches[ptidx].queryIdx](1),
  				     queryNode->feature_locations_3d_[point_matches[ptidx].queryIdx](2));
		
		cv::Point3d train_pt(trainNode->feature_locations_3d_[point_matches[ptidx].trainIdx](0),
  				     trainNode->feature_locations_3d_[point_matches[ptidx].trainIdx](1),
  				     trainNode->feature_locations_3d_[point_matches[ptidx].trainIdx](2));
		
		const cv::DMatch& m = line_matches[i];
		cv::Point3d query_prj = projectPt3d2Ln3d_2 (query_pt, queryNode->lines[m.queryIdx].line3d.A, queryNode->lines[m.queryIdx].line3d.B);
		cv::Point3d train_prj = projectPt3d2Ln3d_2 (train_pt, trainNode->lines[m.trainIdx].line3d.A, trainNode->lines[m.trainIdx].line3d.B);
  		
  		Eigen::Vector3f from(query_prj.x, query_prj.y, query_prj.z);
  		Eigen::Vector3f to(train_prj.x, train_prj.y, train_prj.z);
  		if(isnan(from(2)) || isnan(to(2))) 
			continue;
  		float weight =1/(abs(to[2]) + abs(from[2]));
  		tfc.add(from, to, weight);
	}
	
	//// add points to the tfc
	for(int i=0; i<point_matches.size();++i) {
		const cv::DMatch& m = point_matches[i];
		Eigen::Vector3f from = queryNode->feature_locations_3d_[m.queryIdx].head<3>();
  		Eigen::Vector3f to = trainNode->feature_locations_3d_[m.trainIdx].head<3>();
  		if(isnan(from(2)) || isnan(to(2)))
  			continue;
  		float weight = 1.0;
  		weight =1/(abs(to[2]) + abs(from[2]));
  		tfc.add(from, to, weight);
  	}

	if(tfc.getNoOfSamples()<3)
		valid = false;
  	else
  		valid = true;
	//get relative movement from samples
  	return tfc.getTransformation().matrix();
}


Eigen::Isometry3d getTransform_Line_svd(const Frame* queryNode, const Frame* trainNode, 
				      const std::vector<cv::DMatch>& matches,
				      bool& valid)
// Note: the result transforms a point from the queryNode's CS to the trainNode's CS
{
	if(matches.size()<2) {
		valid = false;
		return Eigen::Isometry3d::Identity();
	}
	vector<RandomLine3d>  query(matches.size()), train(matches.size());
	for(int i=0; i<matches.size(); ++i) {
		query[i] = queryNode->lines[matches[i].queryIdx].line3d;
		train[i] = trainNode->lines[matches[i].trainIdx].line3d;
	}
	cv::Mat R, t;
	valid = computeRelativeMotion_svd (query, train, R, t);
	Eigen::Matrix4f tf;
	tf << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
	      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
	      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2),
	      0, 0, 0, 1;
	
	Eigen::Matrix3d rot;
	rot << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
	      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
	      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
	
	Eigen::AngleAxisd v;
	v.fromRotationMatrix(rot);
	
	Eigen::Isometry3d res = Eigen::Isometry3d::Identity();
	res.rotate(v);
	res.pretranslate(Eigen::Vector3d(t.at<double>(0),t.at<double>(1),t.at<double>(2)));

	return res;
}


double errorFunction(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2,
                      const Eigen::Matrix4d& transformation)
{
	//Take from paramter_server or cam info
	static const double cam_angle_x = 58.0/180.0*M_PI;
	static const double cam_angle_y = 45.0/180.0*M_PI;
	static const double cam_resol_x = 640;
	static const double cam_resol_y = 480;
	static const double raster_stddev_x = 3*tan(cam_angle_x/cam_resol_x);  //5pix stddev in x
	static const double raster_stddev_y = 3*tan(cam_angle_y/cam_resol_y);  //5pix stddev in y
	static const double raster_cov_x = raster_stddev_x * raster_stddev_x;
	static const double raster_cov_y = raster_stddev_y * raster_stddev_y;
	static const bool 	use_error_shortcut = true;

	bool nan1 = isnan(x1(2));
	bool nan2 = isnan(x2(2));
	if(nan1||nan2){
		return std::numeric_limits<double>::max();
	}
	Eigen::Vector4d x_1 = x1.cast<double>();
	Eigen::Vector4d x_2 = x2.cast<double>();

	Eigen::Matrix4d tf_21 = transformation.inverse();

	Eigen::Vector3d mu_1 = x_1.head<3>();
	Eigen::Vector3d mu_2 = x_2.head<3>();
	Eigen::Vector3d mu_1_in_frame_2 = (tf_21 * x_1).head<3>(); // μ₁⁽²⁾  = T₁₂ μ₁⁽¹⁾  
	//New Shortcut to determine clear outliers
	if(use_error_shortcut)
	{
		double delta_sq_norm = (mu_1_in_frame_2 - mu_2).squaredNorm();
		double sigma_max_1 = std::max(raster_cov_x, depth_covariance(mu_1(2)));//Assuming raster_cov_x and _y to be approx. equal
		double sigma_max_2 = std::max(raster_cov_x, depth_covariance(mu_2(2)));//Assuming raster_cov_x and _y to be approx. equal
		if(delta_sq_norm > 2.0 * (sigma_max_1+sigma_max_2)) //Factor 3 for mahal dist should be gotten from caller
		{
			return std::numeric_limits<double>::max();
		}
	} 

	Eigen::Matrix3d rotation_mat = tf_21.block(0,0,3,3);

	//Point 1
	Eigen::Matrix3d cov1 = Eigen::Matrix3d::Zero();
	cov1(0,0) = 1 * raster_cov_x * mu_1(2); //how big is 1px std dev in meter, depends on depth
	cov1(1,1) = 1 * raster_cov_y * mu_1(2); //how big is 1px std dev in meter, depends on depth
	if(nan1) cov1(2,2) = 1e9; //stddev for unknown: should be within 100m
	else     cov1(2,2) = depth_covariance(mu_1(2));

	//Point2
	Eigen::Matrix3d cov2 = Eigen::Matrix3d::Zero();
	cov2(0,0) = 1 * raster_cov_x* mu_2(2); //how big is 1px std dev in meter, depends on depth
	cov2(1,1) = 1 * raster_cov_y* mu_2(2); //how big is 1px std dev in meter, depends on depth
	if(nan2) cov2(2,2) = 1e9; 			   //stddev for unknown: should be within 100m
	else     cov2(2,2) = depth_covariance(mu_2(2));

	Eigen::Matrix3d cov1_in_frame_2 = rotation_mat.transpose() * cov1 * rotation_mat;//Works since the cov is diagonal => Eig-Vec-Matrix is Identity

	// Δμ⁽²⁾ =  μ₁⁽²⁾ - μ₂⁽²⁾
	Eigen::Vector3d delta_mu_in_frame_2 = mu_1_in_frame_2 - mu_2;
	if(std::isnan(delta_mu_in_frame_2(2))){
		delta_mu_in_frame_2(2) = 0.0;//Hack: set depth error to 0 if NaN 
	}
	// Σc = (Σ₁ + Σ₂)
	Eigen::Matrix3d cov_mat_sum_in_frame_2 = cov1_in_frame_2 + cov2;     
	//ΔμT Σc⁻¹Δμ  
	//double sqrd_mahalanobis_distance = delta_mu_in_frame_2.transpose() * cov_mat_sum_in_frame_2.inverse() * delta_mu_in_frame_2;
	double sqrd_mahalanobis_distance = delta_mu_in_frame_2.transpose() *cov_mat_sum_in_frame_2.ldlt().solve(delta_mu_in_frame_2);
	
	if(!(sqrd_mahalanobis_distance >= 0.0))
	{
		return std::numeric_limits<double>::max();
	}
	return sqrd_mahalanobis_distance;
}


Eigen::Isometry3d getTransform_PtsLines_ransac (const Frame* queryNode, const Frame* trainNode, 
			  const std::vector<cv::DMatch> all_point_matches,
			  const std::vector<cv::DMatch> all_line_matches,
			  std::vector<cv::DMatch>& output_point_inlier_matches,
			  std::vector<cv::DMatch>& output_line_inlier_matches,
			  Eigen::Matrix4f& ransac_tf, 
			  float& inlier_rmse,  SystemParameters sysPara)
// input: 3d point matches + 3d line matches
// output: rigid transform between two frames
{
	
	Eigen::Isometry3d resT;
	int nPt = all_point_matches.size();
	int nLn = all_line_matches.size();

	int line_weight = 1;

	vector<int> pt_indexes(nPt);
	vector<int> ln_indexes(nLn);
	for(int i=0; i<pt_indexes.size(); ++i)  pt_indexes[i] = i;
	for(int i=0; i<ln_indexes.size(); ++i)  ln_indexes[i] = i;
	
	int minPtSetSize = min(2, nPt);
	int minLnSetSize = min(2, nLn);
	double ln_angle_thres_deg = sysPara.line3d_angle_relmotion;  //10
	double mahdist4inlier = sysPara.max_mah_dist_for_inliers;    //3

	std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > train_lines_A(nLn), train_lines_B(nLn), query_lines_A(nLn), query_lines_B(nLn);
	for(int i=0; i<nLn; ++i) 
	{
	    cv::DMatch m = all_line_matches[i];
	    train_lines_A[i] = (Eigen::Vector4f(trainNode->lines[m.trainIdx].line3d.A.x, trainNode->lines[m.trainIdx].line3d.A.y, trainNode->lines[m.trainIdx].line3d.A.z, 1));
	    train_lines_B[i] = (Eigen::Vector4f(trainNode->lines[m.trainIdx].line3d.B.x, trainNode->lines[m.trainIdx].line3d.B.y, trainNode->lines[m.trainIdx].line3d.B.z, 1));
	    query_lines_A[i] = (Eigen::Vector4f(queryNode->lines[m.queryIdx].line3d.A.x, queryNode->lines[m.queryIdx].line3d.A.y, queryNode->lines[m.queryIdx].line3d.A.z, 1));
	    query_lines_B[i] = (Eigen::Vector4f(queryNode->lines[m.queryIdx].line3d.B.x, queryNode->lines[m.queryIdx].line3d.B.y, queryNode->lines[m.queryIdx].line3d.B.z, 1));
	} 

	int iter = 0; 	
	float sum_squared_error = 1e9;
	vector<cv::DMatch> max_point_inlier_set, max_line_inlier_set;
	Eigen::Matrix4f tf_best;
	
	//Start Ransac
	while (iter < 50) 
	{
		iter++;
		std::vector<cv::DMatch> ptMch, lnMch;
		random_unique(pt_indexes.begin(), pt_indexes.end(), minPtSetSize);
		for(int i=0; i<minPtSetSize; ++i) 
		{ 
			ptMch.push_back(all_point_matches[pt_indexes[i]]);
		}
		
		
		random_unique(ln_indexes.begin(), ln_indexes.end(), minLnSetSize);
		for(int i=0; i<minLnSetSize; ++i) 
		{
			lnMch.push_back(all_line_matches[ln_indexes[i]]);
		}
		
		Eigen::Matrix4f tf;
		
		float sse = 0;
		
		getTransformFromHybridMatchesG2O(queryNode,trainNode,  ptMch, lnMch, tf, 25, sysPara);

		///// evaluate minimal solution /////
		vector<cv::DMatch> inlier_point_matches, inlier_line_matches;
		inlier_point_matches.reserve(nPt); 
		inlier_line_matches.reserve(nLn);
		Eigen::Matrix4d tf_d = tf.cast<double>();
		for(int i=0; i<all_point_matches.size(); ++i)  
		{
			double mah_dist_sq = errorFunction(queryNode->feature_locations_3d_[all_point_matches[i].queryIdx], 
							    trainNode->feature_locations_3d_[all_point_matches[i].trainIdx], tf_d);
			if (mah_dist_sq < mahdist4inlier * mahdist4inlier) 
			{
				inlier_point_matches.push_back(all_point_matches[i]);
				sse += mah_dist_sq; 
			}        	
		}
		for(int i=0; i<all_line_matches.size(); ++i) 
		{
			Eigen::Vector4f qA_tf = tf * query_lines_A[i];
			Eigen::Vector4f qB_tf = tf * query_lines_B[i];		
			cv::Point3d qA_in_train(qA_tf(0),qA_tf(1),qA_tf(2));
			cv::Point3d qB_in_train(qB_tf(0),qB_tf(1),qB_tf(2));
			double mah_dist_a = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndA, qA_in_train, qB_in_train) ;
			double mah_dist_b = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndB, qA_in_train, qB_in_train);			
			if(mah_dist_a < mahdist4inlier && mah_dist_b < mahdist4inlier) 
			{
				inlier_line_matches.push_back(all_line_matches[i]);
				sse += mah_dist_a * mah_dist_a + mah_dist_b * mah_dist_b;
			}		
		

			double dist = 0.5*dist3d_pt_line (qA_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B)
				    + 0.5*dist3d_pt_line (qB_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B);
			cv::Point3d l1 = qA_in_train - qB_in_train;
			cv::Point3d l2 = trainNode->lines[all_line_matches[i].trainIdx].line3d.A - trainNode->lines[all_line_matches[i].trainIdx].line3d.B;			
			double angle = 180*acos(abs(l1.dot(l2)/cv::norm(l1)/cv::norm(l2)))/PI; 

			if(dist < sysPara.pt2line3d_dist_relmotion && angle < sysPara.line3d_angle_relmotion) 
			{
				inlier_line_matches.push_back(all_line_matches[i]);
				sse += mah_dist_a * mah_dist_a + mah_dist_b * mah_dist_b;
			}
		}
		if(inlier_point_matches.size() + line_weight * inlier_line_matches.size() 
			> max_point_inlier_set.size() + line_weight * max_line_inlier_set.size()) 
		{
			max_point_inlier_set = inlier_point_matches;
			max_line_inlier_set  = inlier_line_matches;
			tf_best = tf;
			sum_squared_error = sse;
		}
	}

	cout<<"RANSAC:"<<max_point_inlier_set.size()<<" "<<max_line_inlier_set.size()<<endl;
	
	///////// refine solution /////////
	vector<cv::DMatch> refined_point_inliers, refined_line_inliers ;
	Eigen::Matrix4f refined_tf = tf_best;
	int tmp_best_line_mah;
	resT = getTransformFromHybridMatchesG2O (queryNode,trainNode,  max_point_inlier_set, max_line_inlier_set, refined_tf, 25, sysPara);
	double refined_rmse = sqrt(sum_squared_error/(max_point_inlier_set.size() + max_line_inlier_set.size()));

	//return resT;

	for(int iter = 0; iter <20; ++iter) {
		vector<cv::DMatch> tmp_pt_inliers, tmp_ln_inliers;
		tmp_pt_inliers.reserve(all_point_matches.size()); tmp_ln_inliers.reserve(all_line_matches.size());
		double tmp_sse = 0;
		
		Eigen::Matrix4d refined_tf_d = refined_tf.cast<double>();
        for(int i=0; i<all_point_matches.size(); ++i)  {
        	double mah_dist_sq = errorFunction(queryNode->feature_locations_3d_[all_point_matches[i].queryIdx], 
        		                             trainNode->feature_locations_3d_[all_point_matches[i].trainIdx], refined_tf_d);
        	
        	if (mah_dist_sq < mahdist4inlier * mahdist4inlier) {
        		tmp_pt_inliers.push_back(all_point_matches[i]);
        		tmp_sse += mah_dist_sq; 
        	}        	
        }

        int tmp_line_mah_inlier = 0;
        for(int i=0; i<all_line_matches.size(); ++i) {
        	Eigen::Vector4f qA_tf = refined_tf * query_lines_A[i];
        	Eigen::Vector4f qB_tf = refined_tf * query_lines_B[i];		
			cv::Point3d qA_in_train(qA_tf(0),qA_tf(1),qA_tf(2)), qB_in_train(qB_tf(0),qB_tf(1),qB_tf(2));
			double mah_dist_a = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndA, qA_in_train, qB_in_train); 
			double mah_dist_b = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndB, qA_in_train, qB_in_train);
			if(mah_dist_a < mahdist4inlier && mah_dist_b < mahdist4inlier) {
				tmp_line_mah_inlier++;
				tmp_ln_inliers.push_back(all_line_matches[i]);
				tmp_sse += mah_dist_a * mah_dist_a + mah_dist_b * mah_dist_b;

			}					
			
			double dist = 0.5*dist3d_pt_line (qA_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B)
						+ 0.5*dist3d_pt_line (qB_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B);
			cv::Point3d l1 = qA_in_train - qB_in_train, 
				l2 = trainNode->lines[all_line_matches[i].trainIdx].line3d.A - trainNode->lines[all_line_matches[i].trainIdx].line3d.B;			
			double angle = 180*acos(abs(l1.dot(l2)/cv::norm(l1)/cv::norm(l2)))/PI; 

			if(dist < sysPara.pt2line3d_dist_relmotion && angle < sysPara.line3d_angle_relmotion) {
				tmp_ln_inliers.push_back(all_line_matches[i]);
				tmp_sse += mah_dist_a * mah_dist_a+ mah_dist_b * mah_dist_b;
			}
	
        }
        if( tmp_pt_inliers.size() + tmp_ln_inliers.size() * sysPara.line_match_number_weight 
              > refined_point_inliers.size() + refined_line_inliers.size() * sysPara.line_match_number_weight)
		{
           	tmp_best_line_mah = tmp_line_mah_inlier;
        	refined_point_inliers = tmp_pt_inliers;
        	refined_line_inliers = tmp_ln_inliers;
        	refined_rmse = sqrt(tmp_sse/(tmp_pt_inliers.size() + tmp_ln_inliers.size()));       
        	resT = getTransformFromHybridMatchesG2O (queryNode,trainNode,refined_point_inliers, refined_line_inliers, refined_tf, 20, sysPara);

        } else
        	break;

   	}

	output_point_inlier_matches = refined_point_inliers;
	output_line_inlier_matches = refined_line_inliers;
	inlier_rmse = refined_rmse;
	ransac_tf = refined_tf;
	return resT;
} 