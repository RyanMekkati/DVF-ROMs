#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace cv;
using std::vector;
using std::string;

void meshgrid(const Range& xr, const Range& yr, Mat& outX, Mat& outY) {
    vector<float> vx, vy;
    for (int i = xr.start; i < xr.end; ++i) vx.push_back((float)i);
    for (int j = yr.start; j < yr.end; ++j) vy.push_back((float)j);
    Mat X(vx), Y(vy);
    repeat(X.reshape(1,1), (int)vy.size(), 1, outX);
    repeat(Y.reshape(1,(int)vy.size()), 1, (int)vx.size(), outY);
}

void expMapSVF(const Mat& vx, const Mat& vy, Mat& phiX, Mat& phiY, int S=6) {
    CV_Assert(vx.size() == vy.size() && vx.type() == CV_32F);
    float scale = 1.0f / float(1 << S);
    Mat vxs = vx * scale, vys = vy * scale;
    Mat gridX, gridY;
    meshgrid(Range(0, vx.cols), Range(0, vx.rows), gridX, gridY);
    gridX.convertTo(gridX, CV_32F);
    gridY.convertTo(gridY, CV_32F);
    phiX = gridX + vxs;
    phiY = gridY + vys;
    Mat tmpX, tmpY;
    for (int i = 0; i < S; ++i) {
        remap(phiX, tmpX, phiX, phiY, INTER_LINEAR);
        remap(phiY, tmpY, phiX, phiY, INTER_LINEAR);
        tmpX.copyTo(phiX);
        tmpY.copyTo(phiY);
    }
}

void extractDVF(const Mat& phiX, const Mat& phiY, Mat& ux, Mat& uy) {
    Mat gridX, gridY;
    meshgrid(Range(0, phiX.cols), Range(0, phiX.rows), gridX, gridY);
    gridX.convertTo(gridX, CV_32F);
    gridY.convertTo(gridY, CV_32F);
    ux = phiX - gridX;
    uy = phiY - gridY;
}

void drawSparseFlow(const Mat& ux, const Mat& uy, const Mat& base, Mat& out,
                    int step = 16, float scale = 5.0f) {
    cvtColor(base, out, COLOR_GRAY2BGR);
    for (int y = step/2; y < ux.rows; y += step) {
        for (int x = step/2; x < ux.cols; x += step) {
            Point2f p0((float)x, (float)y);
            Point2f v(ux.at<float>(y,x), uy.at<float>(y,x));
            Point2f p1(p0.x + v.x*scale, p0.y + v.y*scale);
            line(out, p0, p1, Scalar(0,0,255), 1, LINE_AA);
            const float angle = CV_PI/6, len = 4;
            float theta = atan2(p1.y - p0.y, p1.x - p0.x);
            Point2f h1((float)(cos(theta+CV_PI-angle)*len + p1.x),
                       (float)(sin(theta+CV_PI-angle)*len + p1.y));
            Point2f h2((float)(cos(theta+CV_PI+angle)*len + p1.x),
                       (float)(sin(theta+CV_PI+angle)*len + p1.y));
            line(out, p1, h1, Scalar(0,0,255), 1, LINE_AA);
            line(out, p1, h2, Scalar(0,0,255), 1, LINE_AA);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_png_folder> <output_folder>\n";
        return -1;
    }
    string inDir = argv[1], outDir = argv[2];
    fs::create_directories(outDir + "/flow");
    fs::create_directories(outDir + "/dvf");
    vector<string> files;
    for (auto& e : fs::directory_iterator(inDir))
        if (e.path().extension() == ".png")
            files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if (files.size() < 2) {
        std::cerr << "Need at least two PNGs\n";
        return -1;
    }

    // Video Writer for output
    Mat first = imread(files[0], IMREAD_GRAYSCALE);
    Size frameSize = first.size();
    VideoWriter writer(outDir + "/LDDMM_output.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       10.0, frameSize, true);
    if (!writer.isOpened()) {
        std::cerr << "Cannot open video writer at " << outDir + "/LDDMM_output.avi" << std::endl;
        return -1;
    }

    for (size_t i = 0; i + 1 < files.size(); ++i) {
        Mat I0 = imread(files[i], IMREAD_GRAYSCALE);
        Mat I1 = imread(files[i+1], IMREAD_GRAYSCALE);
        if (I0.empty() || I1.empty()) continue;
        Mat flow;
        calcOpticalFlowFarneback(I0, I1, flow,
                                 0.5, 3, 15, 3, 5, 1.2, 0);
        vector<Mat> ch; split(flow, ch);
        Mat vx, vy;
        ch[0].convertTo(vx, CV_32F);
        ch[1].convertTo(vy, CV_32F);
        Mat phiX, phiY, ux, uy;
        expMapSVF(vx, vy, phiX, phiY, 6);
        extractDVF(phiX, phiY, ux, uy);
        Mat vis;
        drawSparseFlow(ux, uy, I0, vis);
        writer.write(vis);
        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << i;
        imwrite(outDir + "/flow/flow_" + oss.str() + ".png", vis);
        imwrite(outDir + "/dvf/ux_" + oss.str() + ".exr", ux);
        imwrite(outDir + "/dvf/uy_" + oss.str() + ".exr", uy);
        imshow("DVF", vis);
        if (waitKey(30) == 27) break;
    }
    return 0;
}


