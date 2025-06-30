#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <filesystem>           // C++17 filesystem
namespace fs = std::filesystem;

using namespace cv;
using std::vector;
using std::string;

// Draw a sparse quiver of the displacement field (ux,uy)
void drawSparseFlow(const Mat& ux, const Mat& uy,
                    const Mat& base, Mat& out,
                    int step = 16, float scale = 5.0f)
{
    if(base.type()==CV_8UC1){
        cvtColor(base, out, COLOR_GRAY2BGR);
    } else {
        base.copyTo(out);
    }

    for(int y = step/2; y < ux.rows; y += step){
        for(int x = step/2; x < ux.cols; x += step){
            Point2f p0((float)x, (float)y);
            Point2f v(ux.at<float>(y,x), uy.at<float>(y,x));
            Point2f p1 = p0 + v*scale;
            line(out, p0, p1, Scalar(0,0,255), 1, LINE_AA);

            const float angle = CV_PI/6;
            const float len   = 4;
            Point2f dir = p1 - p0;
            float theta = atan2(dir.y, dir.x);
            Point2f h1 = Point2f(cos(theta+CV_PI-angle),
                                 sin(theta+CV_PI-angle)) * len + p1;
            Point2f h2 = Point2f(cos(theta+CV_PI+angle),
                                 sin(theta+CV_PI+angle)) * len + p1;
            line(out, p1, h1, Scalar(0,0,255), 1, LINE_AA);
            line(out, p1, h2, Scalar(0,0,255), 1, LINE_AA);
        }
    }
}

// Utility: meshgrid like MATLAB
void meshgrid(const Range& xr, const Range& yr, Mat& outX, Mat& outY) {
    vector<float> x, y;
    for(int i = xr.start; i < xr.end; ++i) x.push_back((float)i);
    for(int j = yr.start; j < yr.end; ++j) y.push_back((float)j);
    repeat(Mat(x).reshape(1,1), (int)y.size(), 1, outX);
    repeat(Mat(y).reshape(1,(int)y.size()), 1, (int)x.size(), outY);
}

// Exponentiate an SVF (vx,vy) into a diffeo φ via scaling & squaring
void expMapSVF(const Mat& vx, const Mat& vy,
               Mat& phiX, Mat& phiY,
               int S=6)
{
    CV_Assert(vx.size()==vy.size() && vx.type()==CV_32F);
    float scale = 1.f / float(1 << S);
    Mat vxs = vx * scale;
    Mat vys = vy * scale;

    Mat gridX, gridY;
    meshgrid(Range(0, vx.cols), Range(0, vx.rows), gridX, gridY);
    gridX.convertTo(gridX, CV_32F);
    gridY.convertTo(gridY, CV_32F);

    phiX = gridX + vxs;
    phiY = gridY + vys;

    Mat tmpX, tmpY;
    for(int i = 0; i < S; ++i){
        remap(phiX, tmpX, phiX, phiY, INTER_LINEAR);
        remap(phiY, tmpY, phiX, phiY, INTER_LINEAR);
        tmpX.copyTo(phiX);
        tmpY.copyTo(phiY);
    }
}

// Given φ, extract the DVF u = φ − Id
void extractDVF(const Mat& phiX, const Mat& phiY,
                Mat& ux, Mat& uy)
{
    Mat gridX, gridY;
    meshgrid(Range(0, phiX.cols), Range(0, phiX.rows), gridX, gridY);
    gridX.convertTo(gridX, CV_32F);
    gridY.convertTo(gridY, CV_32F);

    ux = phiX - gridX;
    uy = phiY - gridY;
}

int main(int argc, char** argv){
    if(argc < 3){
        std::cerr << "Usage: " << argv[0]
                  << " <path/to/RubberWhale> <outputDir>\n";
        return 1;
    }
    string baseDir   = argv[1];
    string outputDir = argv[2];

    // Create output subdirectories
    fs::create_directories(outputDir + "/flow_frames");
    fs::create_directories(outputDir + "/warp_frames");
    fs::create_directories(outputDir + "/gt_frames");

    // Video writers
    VideoWriter vwFlow, vwWarp, vwGT;
    bool writersInitialized = false;
    int  frameIdx = 0;
    double fps    = 10.0;
    int    fourcc = VideoWriter::fourcc('M','J','P','G');

    // Load the 8-bit sequence
    vector<Mat> seq;
    for(int f = 7; f <= 14; ++f){
        std::ostringstream oss;
        oss << baseDir << "/frame"
            << std::setw(2) << std::setfill('0') << f << ".png";
        Mat I = imread(oss.str(), IMREAD_GRAYSCALE);
        if(I.empty()){
            std::cerr << "Failed to load " << oss.str() << "\n";
            return -1;
        }
        seq.push_back(I);
    }

    // Process every pair
    for(int i = 0; i+1 < (int)seq.size(); ++i){
        Mat I0 = seq[i], I1 = seq[i+1];

        // Optical flow → SVF
        Mat flow; calcOpticalFlowFarneback(
            I0, I1, flow, 0.5,3,15,3,5,1.2,0
        );
        vector<Mat> ch; split(flow, ch);
        Mat vx = ch[0], vy = ch[1];

        // Exponentiate → φ, extract DVF
        Mat phiX, phiY, ux, uy;
        expMapSVF(vx, vy, phiX, phiY, 6);
        extractDVF(phiX, phiY, ux, uy);

        // Visualize
        Mat flowVis, I0warp, I1bgr, warpBGR;
        drawSparseFlow(ux, uy, I0, flowVis);
        remap(I0, I0warp, phiX, phiY,
              INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        cvtColor(I1,    I1bgr,  COLOR_GRAY2BGR);
        cvtColor(I0warp,warpBGR,COLOR_GRAY2BGR);

        // Initialize writers once
        if(!writersInitialized){
            Size sz = flowVis.size();
            vwFlow.open(outputDir + "/dvf.avi",  fourcc, fps, sz);
            vwWarp.open(outputDir + "/warp.avi", fourcc, fps, sz);
            vwGT  .open(outputDir + "/gt.avi",   fourcc, fps, sz);
            writersInitialized = true;
        }

        // Save to video
        vwFlow.write(flowVis);
        vwWarp.write(warpBGR);
        vwGT .write(I1bgr);

        // Save to PNG sequence
        std::ostringstream fn;
        fn << std::setw(3) << std::setfill('0') << frameIdx;
        imwrite(outputDir + "/flow_frames/flow_" + fn.str() + ".png",   flowVis);
        imwrite(outputDir + "/warp_frames/warp_" + fn.str() + ".png",   warpBGR);
        imwrite(outputDir + "/gt_frames/gt_"     + fn.str() + ".png",   I1bgr);

        frameIdx++;

        // (Optional) display
        imshow("DVF on original", flowVis);
        imshow("Warped previous frame", warpBGR);
        imshow("Next frame (ground truth)", I1bgr);
        if(waitKey(30) == 27) break;
    }

    // Release
    vwFlow.release();
    vwWarp.release();
    vwGT .release();

    std::cout << "Saved videos under " << outputDir << "/{dvf,warp,gt}.avi\n"
              << "and frames under " << outputDir << "/{flow,warp,gt}_frames/\n";
    return 0;
}

