#ifndef G2O_VERTEX_LINE6D_H_
#define G2O_VERTEX_LINE6D_H_

#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/stuff/opengl_wrapper.h"
#include <cstdio>
#include <typeinfo>

using namespace Eigen;
namespace Eigen{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;
}

namespace g2o {
  /**
   * \brief Vertex for a tracked point in space
   */
	class VertexLineEndpts : public BaseVertex<6, Eigen::Vector6d>
	{
	public:
		//EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		VertexLineEndpts() {}
		virtual bool read(std::istream& is);
		virtual bool write(std::ostream& os) const;

		virtual void setToOriginImpl() { _estimate.fill(0.); }

		virtual void oplusImpl(const double* update_) {
			Map<const Vector6d> update(update_);
			_estimate += update;
		}

		virtual bool setEstimateDataImpl(const double* est){
			Map<const Vector6d> _est(est);
			_estimate = _est;
			return true;
		}

		virtual bool getEstimateData(double* est) const{
			Map<Vector6d> _est(est);
			_est = _estimate;
			return true;
		}

		virtual int estimateDimension() const {
			return 6;
		}

		virtual bool setMinimalEstimateDataImpl(const double* est){
			_estimate = Map<const Vector6d>(est);
			return true;
		}

		virtual bool getMinimalEstimateData(double* est) const{
			Map<Vector6d> v(est);
			v = _estimate;
			return true;
		}

		virtual int minimalEstimateDimension() const {
			return 6;
		}

	};

	class VertexLineEndptsWriteGnuplotAction: public WriteGnuplotAction
	{
	public:
		VertexLineEndptsWriteGnuplotAction();
		virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_ );
	};

}
#endif