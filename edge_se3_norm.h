#ifndef G2O_EDGE_SE3_LINE_ENDPTS_H_
#define G2O_EDGE_SE3_LINE_ENDPTS_H_
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"
#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
#include "g2o/stuff/opengl_wrapper.h"

#include "Eigen/src/SVD/JacobiSVD.h"
#include <iostream>


namespace g2o {
	using namespace Eigen;
	class EdgeSE3Norm : public BaseBinaryEdge<3, Vector3D, VertexSE3, VertexPointXYZ>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		EdgeSE3Norm();
		virtual bool read(std::istream& is);
		virtual bool write(std::ostream& os) const;
		// return the error estimate as a 3-vector
		void computeError();

		virtual void setMeasurement(const Vector3D& m){
			_measurement = m;
		}

		virtual bool setMeasurementData(const double* d){
			Eigen::Map<const Vector3D> v(d);
			_measurement = v;
			return true;
		}

		virtual bool getMeasurementData(double* d) const{
			Eigen::Map<Vector3D> v(d);
			v=_measurement;
			return true;
		}

		virtual int measurementDimension() const {return 3;}

		virtual bool setMeasurementFromState() ;

		virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& from, 
						OptimizableGraph::Vertex* to) { 
			(void) to; 
			return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
		}

		virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);
		
		const ParameterSE3Offset* offsetParameter() { return offsetParam; }

	private:
		Eigen::Matrix<double,3,9> J; // jacobian before projection
		ParameterSE3Offset* offsetParam;
		CacheSE3Offset* cache;
		virtual bool resolveCaches();
	};

}
#endif