#include "base.h"
#include "utils.h"
#include <random>
using namespace std;

int main()
{
    SystemParameters sysPara;
    Camera camera;
    camera.fx = 517.306408;
    camera.fy = 516.469215;
    camera.cx = 318.643040;
    camera.cy = 255.313989;
    camera.scale = 1000.0;

    Frame f1(0.0, "../data/img1.png","../data/depth1.png",camera);
    Frame f2(0.0, "../data/img4.png","../data/depth4.png",camera);
    
    
    
 

    return 0;
}

