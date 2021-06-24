#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/optflow.hpp>
#include "opt_flow.cpp"
#include <sys/stat.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    string filename = "/coursework/data/Cohort44_Large_Light5.mp4";

    opt_flow(filename);



    return 0;
}