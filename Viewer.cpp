#include "Viewer.h"

Viewer::Viewer()
{
	mImageWidth=640;
	mImageHeight=480;
	
	mViewpointX=10;
	mViewpointY=10;
	mViewpointZ=10;
	mViewpointF=500;
	
// 	mViewpointX=0;
// 	mViewpointY=-0.7;
// 	mViewpointZ=-1.8;
// 	mViewpointF=500;
	
}

void Viewer::run()
{
	pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer",1024,768);
	// 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
	
	// Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	//pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
	
	// Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
	
	pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
	
	while(1)
	{
		// Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);

		// Render OpenGL Cube
		// pangolin::glDrawColouredCube();
		pangolin::glDrawAxis(3);

		//点的创建
		glPointSize(10.0f);
		glBegin(GL_POINTS);
		glColor3f(1.0,1.0,1.0);
		glVertex3f(0.0f,0.0f,0.0f);
		glVertex3f(1,0,0);
		glVertex3f(-2,2,0);
		glEnd();

		//把下面的点都做一次旋转变换
		glPushMatrix();
		//col major
		std::vector<GLfloat > Twc = {1,0,0,0, 0,1,0,0 , 0,0,1,0 ,0,2,5,1};
		glMultMatrixf(Twc.data());

		//直线的创建
		const float w = 2;
		const float h = w*0.75;
		const float z =- w*0.6;
		glLineWidth(2);
		glColor3f(1.0,0,0);
		glBegin(GL_LINES);

		glVertex3f(0,0,0);
		glVertex3f(w,h,z);
		glVertex3f(0,0,0);
		glVertex3f(w,-h,z);
		glVertex3f(0,0,0);
		glVertex3f(-w,-h,z);
		glVertex3f(0,0,0);
		glVertex3f(-w,h,z);

		glVertex3f(w,h,z);
		glVertex3f(w,-h,z);

		glVertex3f(-w,h,z);
		glVertex3f(-w,-h,z);

		glVertex3f(-w,h,z);
		glVertex3f(w,h,z);

		glVertex3f(-w,-h,z);
		glVertex3f(w,-h,z);
		glEnd();

		glPopMatrix();

		// Swap frames and Process Events
		pangolin::FinishFrame();
	}
}


