#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

int opt_flow(const string& filename)
{
    VideoCapture capture(filename);
    if (!capture.isOpened()) {
        cerr << "the file can't be opened!" << endl;
        return 0;
    }

    vector<Scalar> randomColors;
    RNG rng;
    for (int i = 0; i < 200; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        randomColors.push_back(Scalar(r, g, b));
    }

    Mat firstFrame;
    vector<Point2f> firstPoints, currPoints;

    capture >> firstFrame;
    Mat firstFrameGray;
    cvtColor(firstFrame, firstFrameGray, COLOR_BGR2GRAY);

    goodFeaturesToTrack(firstFrameGray, firstPoints, 200, 0.1, 1, Mat(), 8, false, 0.04);

    Mat linesImg = Mat::zeros(firstFrame.size(), firstFrame.type());

    while (true) {
        Mat currFrame;
        capture >> currFrame;

        if (currFrame.empty())
            break;

        Mat currFrameGray;
        cvtColor(currFrame, currFrameGray, COLOR_BGR2GRAY);
        
        vector<uchar> status;
        vector<float> error;
        TermCriteria stopCriteria = TermCriteria((TermCriteria::COUNT) | (TermCriteria::EPS), 20, 0.002);

        calcOpticalFlowPyrLK(firstFrameGray, currFrameGray, firstPoints, currPoints, status, error, Size(13, 13), 3, stopCriteria);

        vector<Point2f> goodPoints;
        for (uint i = 0; i < firstPoints.size(); i++)
        {
            if (status[i] == 1) {
                goodPoints.push_back(currPoints[i]);
                
                line(linesImg, currPoints[i], firstPoints[i], randomColors[i], 2);
                circle(currFrame, currPoints[i], 3, randomColors[i], -1);
            }
        }

        Mat linedFrame;
        add(currFrame, linesImg, linedFrame);

        imshow("Optical Flow", linedFrame);

        int keyboard = waitKey(25);
        if (keyboard == 'q' || keyboard == 27)
            break;

        firstFrameGray = currFrameGray;
        firstPoints = goodPoints;
    }
}
