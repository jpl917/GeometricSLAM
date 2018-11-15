#include "frame.h"
#include "utils.h"

unsigned long int Frame::nextid=0;
bool Frame::mbInitialFlag=true;
Camera 	Frame::camera;  //fx,fy,cx,cy,scale,k1,k2,p1,p2,k3
cv::Mat Frame::K;
cv::Mat Frame::distCoeffs;
float Frame::mnMinX, Frame::mnMaxX, Frame::mnMinY, Frame::mnMaxY;
ORBextractor* Frame::orbextractor;

SystemParameters sysPara;

#ifdef SEGMENT
PyObject* Frame::pNet;
PyObject* Frame::pFunc;	
#endif

FrameLine::FrameLine(cv::Point2d _p, cv::Point2d _q)
{
    p = _p;
    q = _q;
    l = cvpt2mat(p).cross(cvpt2mat(q));
    haveDepth = false;
    gid = -1;
}

cv::Point2d FrameLine::getGradient(cv::Mat* xGradient, cv::Mat* yGradient)
{
    cv::LineIterator iter(*xGradient, p, q, 8);
    double xSum=0, ySum=0;
    for(int i=0; i<iter.count; i++, iter++)
    {
        xSum += xGradient->at<double>(iter.pos());
        ySum += yGradient->at<double>(iter.pos());
    }
    double len = sqrt(xSum*xSum + ySum*ySum);

    return cv::Point2d(xSum/len, ySum/len);
}

Frame::Frame(double _timestamp, cv::Mat _rgb, cv::Mat _depth, Camera _camera)
{
    id = nextid++;
    timestamp = _timestamp;
    rgb = _rgb.clone();
    oriDepth = _depth.clone();
    if(rgb.channels() == 3){
        cv::cvtColor(rgb, gray, CV_RGB2GRAY);
    }
    else{
        gray = rgb;
    }
        
    oriDepth.convertTo(depth, CV_32F);
	//Mat tmp;
    //bilateralFilter(depth, tmp, 25, 25 * 2, 25 / 2);
	//depth=tmp.clone();
	
#ifdef SEGMENT
		segOK = false;
#endif		
	
    if(mbInitialFlag)
    {
		camera = _camera;
		K = cv::Mat::eye(3,3,CV_64F);
		K.at<double>(0,0) = camera.fx;
		K.at<double>(1,1) = camera.fy;
		K.at<double>(0,2) = camera.cx;
		K.at<double>(1,2) = camera.cy;

		distCoeffs = (cv::Mat_<double>(5,1)<<camera.k1,camera.k2,camera.p1,camera.p2,camera.k3);  
		computeImageBoundary();
		mbInitialFlag=false; 
		
#ifdef SEGMENT
		segmentInit();
#endif
	
		
#ifdef GMS_MATCHER
		orbextractor = new ORBextractor(5000,1.2,2);
		cout<<"Init GMS"<<endl;
#else
		orbextractor = new ORBextractor(1000,1.2,5);
		cout<<"Init Filter"<<endl;
#endif
		
    }
    
#ifdef UNDISTORT
    if(cv::norm(distCoeffs)>1e-5)
       cv::undistort(rgb,rgb,K,distCoeffs);
#endif
    
    
#ifdef USE_LINE
    lineLenThresh=50;
    detectFrameLines(1);
    extractLineDepth();
#endif
    extractORB();
    N=mvKeypoints.size();
    if(mvKeypoints.empty())return;
    undistortKeypoints();
    
    //mpORBVocabulary= new ORBVocabulary();
    //mpORBVocabulary->loadFromTextFile("../Vocabulary/ORBvoc.txt");
}

Frame::Frame(double _timestamp, string  rgb_filename, string depth_filename, Camera _camera)
{
	rgbname=rgb_filename;
    id = nextid++;
    timestamp = _timestamp;
    rgb = cv::imread(rgb_filename);
    oriDepth = cv::imread(depth_filename,CV_LOAD_IMAGE_ANYDEPTH);
    //cout<<oriDepth<<endl;
    
    if(rgb.channels() == 3){
      cv::cvtColor(rgb,gray,CV_BGR2GRAY);
    }
    else{
      gray = rgb;
    }
    oriDepth.convertTo(depth, CV_32F);
	//Mat tmp;
    //bilateralFilter(depth, tmp, 25, 25 * 2, 25 / 2);
	//depth=tmp.clone();
	
#ifdef SEGMENT
		segOK = false;
#endif		
	
	
    if(mbInitialFlag)
    {
		camera = _camera;
		K = cv::Mat::eye(3,3,CV_64F);
		K.at<double>(0,0) = camera.fx;
		K.at<double>(1,1) = camera.fy;
		K.at<double>(0,2) = camera.cx;
		K.at<double>(1,2) = camera.cy;
		distCoeffs = (cv::Mat_<double>(5,1)<<camera.k1,camera.k2,camera.p1,camera.p2,camera.k3); 
		computeImageBoundary();
		mbInitialFlag=false;

#ifdef SEGMENT
		segmentInit();
#endif		

		
#ifdef GMS_MATCHER
		orbextractor = new ORBextractor(5000,1.2,2);
		cout<<"Init GMS"<<endl;
#else
		orbextractor = new ORBextractor(1000,1.2,5);
		cout<<"Init Filter"<<endl;
#endif
		
    }
    
#ifdef UNDISTORT
    if(cv::norm(distCoeffs)>1e-5)
       cv::undistort(rgb,rgb,K,distCoeffs);
#endif
    
    
#ifdef USE_LINE
    lineLenThresh=50;
    detectFrameLines(1);
    extractLineDepth();
#endif
    
    extractORB();
    N=mvKeypoints.size();
    if(mvKeypoints.empty())return;
    undistortKeypoints();
    
    //mpORBVocabulary= new ORBVocabulary();
    //mpORBVocabulary->loadFromTextFile("../Vocabulary/ORBvoc.txt");
}


void Frame::undistortKeypoints()
{
    if(distCoeffs.at<float>(0)==0)
    {
      mvKeypointsUn=mvKeypoints;
      return;
    }
    mvKeypointsUn.reserve(N);
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
      mat.at<float>(i,0)=mvKeypoints[i].pt.x;
      mat.at<float>(i,1)=mvKeypoints[i].pt.y;
    }
    
    mat = mat.reshape(2);
    cv::undistortPoints(mat,mat,K,distCoeffs,cv::Mat(),K);
    mat = mat.reshape(1);
    for(int i=0; i<N; i++)
    {
      cv::KeyPoint kp= mvKeypoints[i];
      kp.pt.x=mat.at<float>(i,0);
      kp.pt.y=mat.at<float>(i,1);
      mvKeypointsUn[i]=kp;
    }
    return ;
}


void Frame::computeImageBoundary()
{
	if(distCoeffs.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; 		mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=rgb.cols; 	mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; 		mat.at<float>(2,1)=rgb.rows;
        mat.at<float>(3,0)=rgb.cols; 	mat.at<float>(3,1)=rgb.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,K,distCoeffs,cv::Mat(),K);
        mat=mat.reshape(1);
		
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else{
        mnMinX = 0.0f;
        mnMaxX = rgb.cols;
        mnMinY = 0.0f;
        mnMaxY = rgb.rows;
    }
}




void Frame::clear()
{
	if(isKeyFrame){
		for(int i = 0;i< lines.size(); i++)
		{
			if(lines[i].haveDepth && lines[i].line3d.covA.rows!=0)
				//clear all points in the line
			lines[i].line3d.pts.clear();
		}
	}
	else
		lines.clear();

	vector<KeyPoint> tmp;
	mvKeypoints.swap(tmp);
}



void Frame::detectFrameLines(int method)
{
    int i;
    if(method == 0)
    {
		IplImage pImg = gray;
		ntuple_list lsdOut = callLsd(&pImg);

		int dim = lsdOut->dim;
		double a,b,c,d;
		lines.reserve(lsdOut->size);
		for(i=0; i<lsdOut->size; i++)  //store the lines
		{
			a=lsdOut->values[i*dim];
			b=lsdOut->values[i*dim+1];
			c=lsdOut->values[i*dim+2];
			d=lsdOut->values[i*dim+3];

			if((a-c)*(a-c)+(b-d)*(b-d)>lineLenThresh*lineLenThresh)
				lines.push_back(FrameLine(cv::Point2d(a,b),cv::Point2d(c,d)));
		}
    }
    else
    {
		int n;
		LS* ls = callEDLines(gray, &n);
		lines.reserve(n);
		for(int i=0; i<n; i++) 
		{
			// store output to lineSegments 
			if ((ls[i].sx-ls[i].ex)*(ls[i].sx-ls[i].ex) +(ls[i].sy-ls[i].ey)*(ls[i].sy-ls[i].ey) 
				> lineLenThresh*lineLenThresh) {
				lines.push_back(FrameLine(cv::Point2d(ls[i].sx,ls[i].sy), cv::Point2d(ls[i].ex,ls[i].ey)));
			}
		}
    }
      
    for(i=0; i<lines.size(); i++)
    {
        lines[i].lid = i;
        lines[i].compLineEq2d();
    }
    //compute the MSLD descriptor
    cv::Mat xGradImg, yGradImg;
    cv::Sobel(gray, xGradImg, CV_64F, 1, 0, 3); //gradient x   1 0 3
    cv::Sobel(gray, yGradImg, CV_64F, 0, 1, 3); //gradient y   0 1 3
   
    for(i=0; i< lines.size();i++)  //0.6 ms/line
    {
        computeMSLD(lines[i],&xGradImg, &yGradImg);
    }

}

// input: depth, lines
// output: lines with 3d info
void Frame::extractLineDepth()
{
    double depth_scaling = Frame::camera.scale;  ////////
    int n_3dln = 0;
    for(int i=0; i<lines.size();i++)  //20-30ms
    {
		lines[i].haveDepth = false;
        double len = cv::norm(lines[i].p-lines[i].q);
        vector<cv::Point3d> pts3d;
        double numSmp = (double)min((int)len, 100); //sample the points of the lines
        pts3d.reserve(numSmp);

        for(int j=0; j<numSmp; j++)
        {
            //pt is in the line
            cv::Point2d pt = lines[i].p * (1-j/numSmp) + lines[i].q * (j/numSmp);
            if(pt.x<0||pt.y<0||pt.x>=depth.cols||pt.y>=depth.rows)continue;
            int row, col;
            if((floor(pt.x) ==pt.x) && floor(pt.y)==pt.y){  //boundary issue
                col = max(int(pt.x-1),0);
                row = max(int(pt.y-1),0);
            }else{
                col=int(pt.x);
                row=int(pt.y);
            }
            double zval = -1;

            if(depth.at<float>(row,col) >= 1e-2){
                zval = depth.at<float>(row,col)/depth_scaling; //in meter
            }
            
            if(zval>0)
            {
                cv::Point2d xy3d = mat2cvpt2d(K.inv()*cvpt2mat(pt))*zval;  //1 Todo
                pts3d.push_back(cv::Point3d(xy3d.x,xy3d.y,zval));	
            }
        }

        if(pts3d.size()<max(10.0, 0.3 * numSmp))continue;
		RandomLine3d tmpLine;		
		vector<RandomPoint3d> rndpts3d;
		rndpts3d.reserve(pts3d.size());
		// compute uncertainty of 3d points
		for(int j=0; j<pts3d.size();++j) 
		{
			rndpts3d.push_back(compPt3dCov(pts3d[j], K));
		}

		tmpLine = extract3dline_mahdist(rndpts3d, sysPara);  
		//tmpLine = extract3dline(pts3d, sysPara);
		
		//cout<<pts3d.size()<<" "<<tmpLine.pts.size()<<" "<<cv::norm(tmpLine.A - tmpLine.B)<<endl;
		
		if(tmpLine.pts.size() > 0.3 * numSmp  &&
		cv::norm(tmpLine.A - tmpLine.B) > 0.05) 
		{
			MLEstimateLine3d (tmpLine, 100);
			lines[i].haveDepth = true;
			lines[i].line3d = tmpLine;
			n_3dln++;
			
			std::vector<RandomPoint3d>().swap(tmpLine.pts);
		}
    }
    //cout<<"Line number:"<<lines.size()<<"    Line having depth:"<<n_3dln<<endl;
}

void Frame::write2file(string fname)
{
    fname = fname + num2str(id) + ".txt";
    ofstream file(fname.c_str());
    for(int i=0; i<lines.size(); ++i) {
		if(lines[i].haveDepth) {

			file<<lines[i].line3d.A.x<<'\t'<<lines[i].line3d.A.y<<'\t'<<lines[i].line3d.A.z<<'\t'
			<<lines[i].line3d.B.x<<'\t'<<lines[i].line3d.B.y<<'\t'<<lines[i].line3d.B.z<<'\n';
		}
    }
    file.close();

}

void Frame::extractORB()
{
   orbextractor->extract(rgb,mvKeypoints,mDescriptors);
   feature_locations_2d_ = mvKeypoints;
   projectKeypointTo3d();

}
    
void Frame::projectKeypointTo3d()
{
    if(feature_locations_3d_.size()) feature_locations_3d_.clear();
    for(int i=0; i< feature_locations_2d_.size(); i++)
    {
		cv::Point2f p2d = feature_locations_2d_[i].pt;
		float d = depth.ptr<float>(int(p2d.y))[int(p2d.x)]/camera.scale;
		float x = (p2d.x-camera.cx)*d/camera.fx;
		float y = (p2d.y-camera.cy)*d/camera.fy;
		//cout<<x<<" "<<y<<" "<<d<<endl;
		feature_locations_3d_.push_back(Eigen::Vector4f(x,y,d,1.0));
    }
    //feature_locations_2d_.resize(feature_locations_3d_.size());
}
 
    
//Point cloud
PointCloud::Ptr Frame::img2cloud()
{
    PointCloud::Ptr cloud(new PointCloud);
    for(int i=0;i<depth.rows;i+=3)
    {
		for(int j=0;j<depth.cols;j+=3)
		{
			double d=(double)depth.ptr<float>(i)[j]/camera.scale;
			if(d <= 1e-2||d>=10)continue;
			PointT p;
			p.z= d;
			p.x=(j-camera.cx)*p.z/camera.fx;
			p.y=(i-camera.cy)*p.z/camera.fy;
			
			p.b=rgb.ptr<uchar>(i)[j*3];
			p.g=rgb.ptr<uchar>(i)[j*3+1];
			p.r=rgb.ptr<uchar>(i)[j*3+2];
			

			cloud->points.push_back(p);
		}
    }
    cloud->width=cloud->points.size();
    cloud->height=1;
    cloud->is_dense=false;
    
    return cloud;
}


void Frame::computeBow()
{
  /*
    if(mBowVec.empty()||mFeatureVec.empty())
    {
		vector<cv::Mat> vDescTmp;
		vDescTmp.reserve(mDescriptors.rows);
		for(int i=0; i<mDescriptors.rows;i++)
		{
			vDescTmp.push_back(mDescriptors.row(i));
		}
		
		mpORBVocabulary->transform(vDescTmp,mBowVec,mFeatureVec,4);
    }*/
    
}

void Frame::setPose(Mat Tcw)
{
    mTcw=Tcw.clone();
    mRcw=mTcw.rowRange(0,3).colRange(0,3);
    mtcw=mTcw.rowRange(0,3).col(3);
    mRwc=mRcw.t();
    mOw= -mRwc*mtcw;    //=mtwc
}

#ifdef SEGMENT
void Frame::segmentInit()
{
	Py_Initialize();//ini
	/*call python 0.5s*/
	string path = "/home/jpl/lines/pydensecrf/cmuFcnAlexnet";
	string chdir_cmd = string("sys.path.append(\"") + path + "\")";
	const char* cstr_cmd = chdir_cmd.c_str();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(cstr_cmd);
	
	PyObject* pModule = PyImport_ImportModule("inference1");
	if(!pModule)
	{
		cout<<"pModule Error"<<endl;
		return;
	}
	
	PyObject* pInit = PyObject_GetAttrString(pModule, "ini");
	if(!pInit)
	{
		cout<<"pIni Error"<<endl;
		return;
	}
	pNet = PyObject_CallObject(pInit, NULL);
	if(!pNet)
	{
		cout<<"pNet Error"<<endl;		
		return;
	}

	pFunc = PyObject_GetAttrString(pModule, "seg");
	if(!pFunc)
	{
		cout<<"pFunc Error"<<endl;		
		return;
	}
}


PointT generatePoint(Mat rgb, Mat depth, int i, int j, double _fx , double _fy, double _cx, double _cy, double _s)
{
	
	PointT point;
	double d=(double)depth.ptr<unsigned short>(i)[j]/_s;
	if(d<=1e-3||d>=10) return point;
	point.z =  d;
	point.x = (j - _cx) * d / _fx;
	point.y = (i - _cy) * d / _fy;
	point.b = rgb.ptr<uchar>(i)[j*3];
	point.g = rgb.ptr<uchar>(i)[j*3+1];
	point.r = rgb.ptr<uchar>(i)[j*3+2];
	
	return point;
}

void Frame::segment()
{
	//if(rgbname=="")return;
	/*call segmentation ~50 ms GPU*/
	PyObject* arg0 = pNet;
	PyObject* arg1 = Py_BuildValue("s", rgbname.c_str());    
	PyObject* arg2 = Py_BuildValue("s", "tmp.png");
	PyObject* args = PyTuple_New(3); 
	PyTuple_SetItem(args, 0, arg0);    
	PyTuple_SetItem(args, 1, arg1);
	PyTuple_SetItem(args, 2, arg2);
	
	PyObject* pRet = PyObject_CallObject(pFunc, args);
	PointCloud::Ptr cloud(new PointCloud);
	
	PointT point;

	if(pRet)
	{
		PyArrayObject* in_con = (PyArrayObject*)pRet;
		unsigned char* ptr = (unsigned char*)PyArray_DATA(in_con);
		npy_intp* pdim = PyArray_DIMS(in_con);

		for(int i=0; i<pdim[0]*pdim[1]; i++)
		{
			point = generatePoint(rgb, depth, i/pdim[1],i%pdim[1], camera.fx,camera.fy,camera.cx,camera.cy,camera.scale);
			if(point.z==0)continue;
			if((int)ptr[i]!=255)
			{
				cloud->push_back(point);
			}
			
		}
	}

	//ax+by+cz+d=0
	c.reserve(4);
	if(cloud->points.size()==0)return ;

	/*calculate normal 8ms*/
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (1000);
	seg.setDistanceThreshold (0.01);
	seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);

	segOK=true;
	cout<<"Coefficients: ";
	for(size_t i=0; i<4; i++) 
	{
		c[i]=coefficients->values[i];
		cout<<c[i]<<" ";
	}
	cout<<endl;

}
#endif

void Frame::getEquation(PointCloud::Ptr cloud, vector<double>& c)
{
	//cout<<cloud->size()<<endl;
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (1000);
	seg.setDistanceThreshold (0.01);
	seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);
	//cout<<cloud->size()<<endl;
	c.reserve(4);
	for(size_t i=0; i<4; i++) 
	{	
		c[i]=coefficients->values[i];
		cout<<c[i]<<" ";
	}
	cout<<endl;
	return;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Frame::planeExtraction()
{
	PointCloud::Ptr cloud=img2cloud();
	pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree = 
		boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA> > (new pcl::search::KdTree<pcl::PointXYZRGBA>);
	pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod (tree);
	normal_estimator.setInputCloud (cloud);
	normal_estimator.setKSearch (100);
	//normal_estimator.setRadiusSearch (0.05);
	normal_estimator.compute (*normals);

	pcl::IndicesPtr indices (new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZRGBA> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0, 10.0);
	pass.filter (*indices);
	//cout<<"size2:"<<cloud->size()<<endl;;

	pcl::RegionGrowing<pcl::PointXYZRGBA, pcl::Normal> reg;
	reg.setMinClusterSize (1000);
	reg.setMaxClusterSize (1000000);
	reg.setSearchMethod (tree);
	reg.setNumberOfNeighbours (30);
	reg.setInputCloud (cloud);
	reg.setIndices (indices);
	reg.setInputNormals (normals);
	reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
	reg.setCurvatureThreshold (0.10);

	std::vector <pcl::PointIndices> clusters;
	reg.extract(clusters);

	std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
	vector<double> c;
	srand(unsigned(time(NULL)));
	//Mat rs= frame.rgb.clone();
	for(int i=0; i<clusters.size(); i++)
	{
		//std::cout << "Cluster " <<i<<" "<<clusters[i].indices.size () << " points." << endl;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr each_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
		//each_plane->resize(clusters[i].indices.size ());
		int r = rand()%255;
		int g = rand()%255;
		int b = 200;
		for (size_t j = 0; j < clusters[i].indices.size (); ++j)
		{

			pcl::PointXYZRGBA point;
			point.x = cloud->points[clusters[i].indices[j]].x;
	  		point.y = cloud->points[clusters[i].indices[j]].y;
	 		point.z = cloud->points[clusters[i].indices[j]].z;
// 			point.r = r;
// 			point.g = g;
// 			point.b = b;
// 			point.a = 0.0;
			point.r = cloud->points[clusters[i].indices[j]].r;
			point.g = cloud->points[clusters[i].indices[j]].g;
			point.b = cloud->points[clusters[i].indices[j]].b;
			point.a = cloud->points[clusters[i].indices[j]].a;
			each_plane->push_back(point);
			cloud_plane->push_back(point);
		}
		getEquation(each_plane,c);
	}
	//pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
	//pcl::visualization::CloudViewer viewer ("Cluster viewer");
	//viewer.showCloud(cloud_plane);
	//while (!viewer.wasStopped ()){}
	
	/**/
	return cloud_plane;
}

void Frame::planeEquation(vector<vector<double>>& equations)
{
	PointCloud::Ptr cloud=img2cloud();
	pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree = 
		boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA> > (new pcl::search::KdTree<pcl::PointXYZRGBA>);
	pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod (tree);
	normal_estimator.setInputCloud (cloud);
	normal_estimator.setKSearch (100);
	//normal_estimator.setRadiusSearch (0.05);
	normal_estimator.compute (*normals);

	pcl::IndicesPtr indices (new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZRGBA> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0, 10.0);
	pass.filter (*indices);
	//cout<<"size2:"<<cloud->size()<<endl;;

	pcl::RegionGrowing<pcl::PointXYZRGBA, pcl::Normal> reg;
	reg.setMinClusterSize (1000);
	reg.setMaxClusterSize (1000000);
	reg.setSearchMethod (tree);
	reg.setNumberOfNeighbours (30);
	reg.setInputCloud (cloud);
	reg.setIndices (indices);
	reg.setInputNormals (normals);
	reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
	reg.setCurvatureThreshold (0.10);

	std::vector <pcl::PointIndices> clusters;
	reg.extract(clusters);

	std::cout << "Cluster:" << clusters.size () << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
	vector<double> c(4);
	srand(unsigned(time(NULL)));
	//Mat rs= frame.rgb.clone();
	for(int i=0; i<clusters.size(); i++)
	{
		//std::cout << "Cluster " <<i<<" "<<clusters[i].indices.size () << " points." << endl;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr each_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
		for (size_t j = 0; j < clusters[i].indices.size (); ++j)
		{
			pcl::PointXYZRGBA point;
			point.x = cloud->points[clusters[i].indices[j]].x;
	  		point.y = cloud->points[clusters[i].indices[j]].y;
	 		point.z = cloud->points[clusters[i].indices[j]].z;
			point.r = cloud->points[clusters[i].indices[j]].r;
			point.g = cloud->points[clusters[i].indices[j]].g;
			point.b = cloud->points[clusters[i].indices[j]].b;
			point.a = cloud->points[clusters[i].indices[j]].a;
			each_plane->push_back(point);
		}
		//cout<<"i:"<<i<<endl;
		getEquation(each_plane,c);
		equations.push_back(c);
	}

	return;
}


void Frame::AHCPlane()
{
// 	cv::Mat depth = cv::imread("data/1depth.png",cv::IMREAD_ANYDEPTH);
//     const float fx = 525;
// 	const float fy = 525;
//     const float cx = 319.5;
//     const float cy = 239.5;
//     const float max_use_range = 10;
	
	MyTimer mytimer;
	mytimer.start();
	
    const float fx = camera.fx;
	const float fy = camera.fy;
    const float cx = camera.cx;
    const float cy = camera.cy;
    const float max_use_range = 10;
	
    cv::Mat_<cv::Vec3f> cloud(depth.rows, depth.cols);
    for(int r=0; r<depth.rows; r++)
    {
        const float* depth_ptr = depth.ptr<float>(r); //unsigned short
        cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
        for(int c=0; c<depth.cols; c++)
        {
            float z = (float)depth_ptr[c]/camera.scale;
            if(z>max_use_range){z=0;}
            pt_ptr[c][0] = (c-cx)/fx*z*1000.0;//m->mm
            pt_ptr[c][1] = (r-cy)/fy*z*1000.0;//m->mm
            pt_ptr[c][2] = z*1000.0;//m->mm
        }
    }

    PlaneFitter pf;
    pf.minSupport = 3000;
    pf.windowWidth = 10;
    pf.windowHeight = 10;
    pf.doRefine = true;

    cv::Mat seg(depth.rows, depth.cols, CV_8UC3);
    OrganizedImage3D Ixyz(cloud);
    pf.run(&Ixyz, 0, &seg);

	mytimer.end();
	
    cv::Mat depth_color;
    depth.convertTo(depth_color, CV_8UC1, 50.0/camera.scale);
    applyColorMap(depth_color, depth_color, cv::COLORMAP_JET);
    cv::imshow("seg",seg);
    cv::imshow("depth",depth_color);
    cv::waitKey(0);
	
	return;
}







