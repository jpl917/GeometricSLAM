#ifndef BASE_H
#define BASE_H

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <cmath>
#include <iomanip>
#include <vector>
#include <ctime>
#include <time.h>
#include <cxcore.h>

#include <tr1/unordered_set>
#include <tr1/unordered_map>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/search/organized.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/don.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>


#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/slam3d/edge_se3_pointxyz_depth.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/core/estimate_propagator.h>
#include <g2o/core/hyper_dijkstra.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/marginal_covariance_cholesky.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/batch_stats.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/stuff/timeutil.h>
#include <g2o/stuff/sparse_helper.h>

#include <armadillo>

#include "Thirdparty/levmar-2.6/levmar.h"
#include "Thirdparty/DBoW2/DBoW2/FORB.h"
#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

//#include <cholmod.h>
using namespace cv;
using namespace std;
using namespace g2o;
using namespace Eigen;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//#define VERBOSE
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;


class MyTimer
{
public:
    timespec t0, t1; 
    MyTimer() {}
    double time_ms;
    double time_s;
    void start() {
		clock_gettime(CLOCK_REALTIME, &t0);
    }
    void end() {
		clock_gettime(CLOCK_REALTIME, &t1);
		time_ms = t1.tv_sec * 1000 + t1.tv_nsec/1000000.0 - (t0.tv_sec * 1000 + t0.tv_nsec/1000000.0);
		time_s = time_ms/1000.0;
		cout<<"Time: "<<time_ms<<" ms"<<endl;
    }
};





#endif
