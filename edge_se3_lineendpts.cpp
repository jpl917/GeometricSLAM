#include "edge_se3_lineendpts.h"
#include "utils.h"

using namespace std;

namespace g2o {
	using namespace std;

	// point to camera projection, monocular
	EdgeSE3LineEndpts::EdgeSE3LineEndpts() : BaseBinaryEdge<6, Vector6d, VertexSE3, VertexLineEndpts>() {
		information().setIdentity();
		J.fill(0);
		J.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
		cache = 0;
		offsetParam = 0;
		resizeParameters(1);
		installParameter(offsetParam, 0, 0); 
	}

	bool EdgeSE3LineEndpts::resolveCaches(){
		ParameterVector pv(1);
		pv[0]=offsetParam;
		resolveCache(cache, (OptimizableGraph::Vertex*)_vertices[0],"CACHE_SE3_OFFSET",pv);
		return cache != 0;
	}


	bool EdgeSE3LineEndpts::read(std::istream& is) {
		int pId;
		is >> pId;
		setParameterId(0, pId);
		// measured endpoints
		Vector6d meas;
		for (int i=0; i<6; i++) is >> meas[i];
		setMeasurement(meas);
		// information matrix is the identity for features, could be changed to allow arbitrary covariances    
		if (is.bad()) {
			return false;
		}
		for ( int i=0; i<information().rows() && is.good(); i++)
			for (int j=i; j<information().cols() && is.good(); j++){
				is >> information()(i,j);
				if (i!=j)
					information()(j,i)=information()(i,j);
			}
			if (is.bad()) {
				//  we overwrite the information matrix
				information().setIdentity();
			} 
			return true;
	}

	bool EdgeSE3LineEndpts::write(std::ostream& os) const {
		os << offsetParam->id() << " ";
		for (int i=0; i<6; i++) os  << measurement()[i] << " ";
		for (int i=0; i<information().rows(); i++)
			for (int j=i; j<information().cols(); j++) {
				os <<  information()(i,j) << " ";
			}
			return os.good();
	}


	void EdgeSE3LineEndpts::computeError() {
		
		VertexLineEndpts *endpts = static_cast<VertexLineEndpts*>(_vertices[1]);
		
		Vector3d ptAw(endpts->estimate()[0],endpts->estimate()[1],endpts->estimate()[2]);
		Vector3d ptBw(endpts->estimate()[3],endpts->estimate()[4],endpts->estimate()[5]);
		
		Vector3d ptA = cache->w2n() * ptAw; // line endpoint tranformed to the camera frame
		Vector3d ptB = cache->w2n() * ptBw;
		
		Vector3d measpt1(_measurement(0),_measurement(1),_measurement(2));
		Vector3d measpt2(_measurement(3),_measurement(4),_measurement(5));
		
		Eigen::Vector3d Ap = endpt_AffnMat.block<3,3>(0,0) * (ptA - measpt1);
		Eigen::Vector3d Bp = endpt_AffnMat.block<3,3>(0,0) * (ptB - measpt1);
		Eigen::Vector3d Bp_Ap = Bp - Ap;
		double t = - Ap.dot(Bp_Ap)/(Bp_Ap.dot(Bp_Ap));
		Vector3d normalized_pt2line_vec1 = Ap+ t*Bp_Ap;

		Eigen::Vector3d Ap2 = endpt_AffnMat.block<3,3>(3,3) * (ptA - measpt2);
		Eigen::Vector3d Bp2 = endpt_AffnMat.block<3,3>(3,3) * (ptB - measpt2);
		Eigen::Vector3d Bp2_Ap2 = Bp2 - Ap2;
		double t2 = - Ap2.dot(Bp2_Ap2)/(Bp2_Ap2.dot(Bp2_Ap2));
		Vector3d normalized_pt2line_vec2 = Ap2+ t2*Bp2_Ap2;

		_error.resize(6);
		_error(0) = normalized_pt2line_vec1(0);
		_error(1) = normalized_pt2line_vec1(1);
		_error(2) = normalized_pt2line_vec1(2);
		_error(3) = normalized_pt2line_vec2(0);
		_error(4) = normalized_pt2line_vec2(1);
		_error(5) = normalized_pt2line_vec2(2);
	}

	bool EdgeSE3LineEndpts::setMeasurementFromState()
	{ 
		//VertexSE3 *cam = static_cast<VertexSE3*>(_vertices[0]);
		//VertexLineEndpts *lpts = static_cast<VertexLineEndpts*>(_vertices[1]);

		VertexLineEndpts *endpts = static_cast<VertexLineEndpts*>(_vertices[1]);
		Vector3d ptAw(endpts->estimate()[0],endpts->estimate()[1],endpts->estimate()[2]);
		Vector3d ptBw(endpts->estimate()[3],endpts->estimate()[4],endpts->estimate()[5]);
		Vector3d ptA = cache->w2n() * ptAw; // line endpoint tranformed to the camera frame
		Vector3d ptB = cache->w2n() * ptBw;
		_measurement.resize(6);
		_measurement(0) = ptA(0);
		_measurement(1) = ptA(1);
		_measurement(2) = ptA(2);
		_measurement(3) = ptB(0);
		_measurement(4) = ptB(1);
		_measurement(5) = ptB(2);
		return true;
	}


	void EdgeSE3LineEndpts::initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* /*to_*/) // ???
	{ // estimate 3d pt world position by cam pose and current meas pt
		(void) from;
		assert(from.size() == 1 && from.count(_vertices[0]) == 1 && "Can not initialize VertexDepthCam position by VertexTrackXYZ");

		VertexSE3 *cam = dynamic_cast<VertexSE3*>(_vertices[0]);
		VertexLineEndpts *point = dynamic_cast<VertexLineEndpts*>(_vertices[1]);
	}

}