#ifndef G2O_EDGE_SE3_LINE_ENDPTS_H_
#define G2O_EDGE_SE3_LINE_ENDPTS_H_
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"
#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
#include "g2o/stuff/opengl_wrapper.h"

#include "Eigen/src/SVD/JacobiSVD.h"
#include "vertex_lineendpts.h"
#include <iostream>


namespace g2o {
	// first two args are the measurement type, second two the connection classes
	using namespace Eigen;
	class EdgeSE3LineEndpts : public BaseBinaryEdge<6, Vector6d, VertexSE3, VertexLineEndpts>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		EdgeSE3LineEndpts();
		virtual bool read(std::istream& is);
		virtual bool write(std::ostream& os) const;
		// return the error estimate as a 3-vector
		void computeError();

		virtual void setMeasurement(const Vector6d& m){
			_measurement = m;
		}

		virtual bool setMeasurementData(const double* d){
			Eigen::Map<const Vector6d> v(d);
			_measurement = v;
			return true;
		}

		virtual bool getMeasurementData(double* d) const{
			Eigen::Map<Vector6d> v(d);
			v=_measurement;
			return true;
		}

		virtual int measurementDimension() const {return 6;}

		virtual bool setMeasurementFromState() ;

		virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& from, 
						OptimizableGraph::Vertex* to) { 
			(void) to; 
			return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
		}

		virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);

		Eigen::Matrix<double,6,6> endptCov;
		Eigen::Matrix<double,6,6> endpt_AffnMat; // to compute mahalanobis dist

	private:
		Eigen::Matrix<double,6,6+6> J; // jacobian before projection
		ParameterSE3Offset* offsetParam;
		CacheSE3Offset* cache;
		virtual bool resolveCaches();
	};

}
#endif