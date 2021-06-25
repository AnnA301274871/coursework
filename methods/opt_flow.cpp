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

    Mat plotImage = Mat(170, 1000, CV_8UC3);
    plotImage.setTo(Scalar(255, 255, 255));
    putText(plotImage, "East", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
    putText(plotImage, "North-East", Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
    putText(plotImage, "North", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
    putText(plotImage, "North-West", Point(10, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
    putText(plotImage, "West", Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
    putText(plotImage, "Sourth-West", Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
    putText(plotImage, "Sourth", Point(10, 140), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
    putText(plotImage, "Sourth-East", Point(10, 160), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));

    vector<float> direct(8, 0.0); 

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

                if (norm(currPoints[i] - firstPoints[i]) < 11) {
                    line(linesImg, currPoints[i], firstPoints[i], randomColors[i], 2);
                    circle(currFrame, currPoints[i], 3, randomColors[i], -1);
                }

                
                float cosAngle = (currPoints[i].x - firstPoints[i].x) / sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                if (cosAngle < -0.9239) {
                    direct[4] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                }
                else {
                    if (cosAngle < -0.3827) {
                        if ((currPoints[i].y - firstPoints[i].y) < 0) {
                            direct[3] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                        }
                        else {
                            direct[5] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                        }
                    }
                    else {
                        if (cosAngle < 0.3827) {
                            if ((currPoints[i].y - firstPoints[i].y) < 0) {
                                direct[2] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                            }
                            else {
                                direct[6] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                            }
                        }
                        else {
                            if (cosAngle < 0.9239) {
                                if ((currPoints[i].y - firstPoints[i].y) < 0) {
                                    direct[1] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                                }
                                else {
                                    direct[7] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                                }
                            }
                            else {
                                direct[0] += sqrt((currPoints[i].x - firstPoints[i].x) * (currPoints[i].x - firstPoints[i].x) + (currPoints[i].y - firstPoints[i].y) * (currPoints[i].y - firstPoints[i].y));
                            }
                        }
                    }
                }
            }
        }

        for (uint i = 0; i < direct.size(); i++) {
            line(plotImage, Point(140, 14 + 20 * i), Point(140 + direct[i] / 10.0, 14 + 20 * i), Scalar::all(0), 15);
        } 

        Mat linedFrame;
        add(currFrame, linesImg, linedFrame);

        imshow("Optical Flow", linedFrame);

        imshow("Chart", plotImage);

        int keyboard = waitKey(25);
        if (keyboard == 'q' || keyboard == 27)
            break;

        firstFrameGray = currFrameGray;
        firstPoints = goodPoints;
    }
}

/*
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

                if (norm(currPoints[i] - firstPoints[i]) < 20) {
                    line(linesImg, currPoints[i], firstPoints[i], randomColors[i], 2);
                    circle(currFrame, currPoints[i], 3, randomColors[i], -1);
                }
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




*/
