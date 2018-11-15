#ifndef MAP_H
#define MAP_H

class Map3d
{
public:
	string			datapath;
	vector<Frame>	frames;
	vector<int>		keyframeIdx;
	vector<LmkLine>	lmklines;
	vector<string>  paths;
	cv::Mat			vel, w;  // linear vol in world coord
	int				lastLoopCloseFrame;
	bool			useConstVel;

	Map3d(){}
	Map3d(string path):datapath(path){}
	Map3d(vector<string> paths_):paths(paths_){}
	void slam(){}	
	void compGT();
#ifdef SLAM_LBA	
	void lba(int numPos=3, int numFrm=5);
	void lba_g2o(int numPos=3, int numFrm=5, int mode = 0);
	void loopclose(){}
	void correctPose(vector<PoseConstraint> pcs);
	void correctAll(vector<vector<int> > lmkGidMatches);
#endif	
};


#endif