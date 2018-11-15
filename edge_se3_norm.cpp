#include "edge_se3_norm.h"
#include "utils.h"

using namespace std;

namespace g2o {
	using namespace std;

	EdgeSE3Norm::EdgeSE3Norm() : BaseBinaryEdge<3, Vector3D, VertexSE3, VertexPointXYZ>() {
		information().setIdentity();
		J.fill(0);
		J.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
		cache = 0;
		offsetParam = 0;
		resizeParameters(1);
		installParameter(offsetParam, 0, 0); 
	}

	bool EdgeSE3Norm::resolveCaches(){
		ParameterVector pv(1);
		pv[0]=offsetParam;
		resolveCache(cache, (OptimizableGraph::Vertex*)_vertices[0],"CACHE_SE3_OFFSET",pv);
		return cache != 0;
	}


	bool EdgeSE3Norm::read(std::istream& is) {
		int pId;
		is >> pId;
		setParameterId(0, pId);
		// measured endpoints
		Vector3D meas;
		for (int i=0; i<3; i++) is >> meas[i];
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

	bool EdgeSE3Norm::write(std::ostream& os) const {
		os << offsetParam->id() << " ";
		for (int i=0; i<3; i++) os  << measurement()[i] << " ";
		for (int i=0; i<information().rows(); i++)
			for (int j=i; j<information().cols(); j++) {
				os <<  information()(i,j) << " ";
			}
			return os.good();
	}


	void EdgeSE3Norm::computeError() {
		VertexSE3 *cam = static_cast<VertexSE3>(_vertices[0]);
		
		VertexPointXYZ *point = static_cast<VertexPointXYZ*>(_vertices[1]);

		Vector3D perr = cache->w2n() * point->estimate();

		// error, which is backwards from the normal observed - calculated
		// _measurement is the measured projection
		_error = perr - _measurement;	
	}
	
	
	bool EdgeSE3Norm::setMeasurementFromState()
	{ 
		// calculate the projection
		VertexPointXYZ *point = static_cast<VertexPointXYZ*>(_vertices[1]);
		
		const Vector3D &pt = point->estimate();
		// SE3OffsetCache* vcache = (SE3OffsetCache*) cam->getCache(_cacheIds[0]);
		// if (! vcache){
		//   cerr << "fatal error in retrieving cache" << endl;
		// }

		Vector3D perr = cache->w2n() * pt;
		_measurement = perr;
		return true;
	}


	void EdgeSE3Norm::initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* /*to_*/) // ???
	{ // estimate 3d pt world position by cam pose and current meas pt
		(void) from;
		assert(from.size() == 1 && from.count(_vertices[0]) == 1 && "Can not initialize VertexDepthCam position by VertexTrackXYZ");

		VertexSE3 *cam = dynamic_cast<VertexSE3*>(_vertices[0]);
		VertexPointXYZ *point = dynamic_cast<VertexPointXYZ*>(_vertices[1]);
		
		//Vector3D p=_measurement;
		//point->setEstimate(cam->estimate() * (offsetParam->offset() * p));
	}

}