#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// --------------------------------------------------
// Sous-échantillonnage (inchangé)
// --------------------------------------------------
Mat downsample(const Mat& input, int factor) {
    Mat output(input.rows, input.cols, input.type());
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            int srcY = min((y / factor) * factor, input.rows - 1);
            int srcX = min((x / factor) * factor, input.cols - 1);
            output.at<float>(y, x) = input.at<float>(srcY, srcX);
        }
    }
    return output;
}

// ——————————————————————————————————————————
// Amplify the warp for visualization
// ——————————————————————————————————————————
void amplifyDisplacement(Mat& phiX, Mat& phiY, float amp)
{
    for(int y = 0; y < phiX.rows; y++){
        for(int x = 0; x < phiX.cols; x++){
            float dx = phiX.at<float>(y,x) - x;
            float dy = phiY.at<float>(y,x) - y;
            phiX.at<float>(y,x) = x + dx * amp;
            phiY.at<float>(y,x) = y + dy * amp;
        }
    }
}

// --------------------------------------------------
// Horn–Schunck (inchangé)
// --------------------------------------------------
void computeHornSchunck(const Mat& I1, const Mat& I2, Mat& u, Mat& v, float alpha, int max_iter) {
    Mat Ix, Iy, It;
    Sobel(I1, Ix, CV_32F, 1, 0, 3);
    Sobel(I1, Iy, CV_32F, 0, 1, 3);
    It = I2 - I1;

    u = Mat::ones(I1.size(), CV_32F);
    v = Mat::zeros(I1.size(), CV_32F);

    for (int iter = 0; iter < max_iter; iter++) {
        Mat u_avg, v_avg;
        blur(u, u_avg, Size(3, 3));
        blur(v, v_avg, Size(3, 3));

        for (int y = 0; y < I1.rows; y++) {
            for (int x = 0; x < I1.cols; x++) {
                float Ex = Ix.at<float>(y, x);
                float Ey = Iy.at<float>(y, x);
                float Et = It.at<float>(y, x);
                float num = Ex * u_avg.at<float>(y, x) +
                            Ey * v_avg.at<float>(y, x) + Et;
                float den = alpha * alpha + Ex*Ex + Ey*Ey;
                float upd = num / den;
                u.at<float>(y, x) = u_avg.at<float>(y, x) - Ex * upd;
                v.at<float>(y, x) = v_avg.at<float>(y, x) - Ey * upd;
            }
        }
    }
}

// --------------------------------------------------
// Exponentiation du flux → map diffeomorphe
// --------------------------------------------------
void exponentiateFlow(const Mat& u, const Mat& v, Mat& phiX, Mat& phiY, int squarings) {
    float scale = 1.f / float(1 << squarings);
    Mat V;
    {
        vector<Mat> ch{u * scale, v * scale};
        merge(ch, V);  // CV_32FC2
    }

    phiX.create(u.size(), CV_32FC1);
    phiY.create(u.size(), CV_32FC1);
    for (int y = 0; y < u.rows; y++) {
        for (int x = 0; x < u.cols; x++) {
            Vec2f vv = V.at<Vec2f>(y, x);
            phiX.at<float>(y, x) = x + vv[0];
            phiY.at<float>(y, x) = y + vv[1];
        }
    }

    for (int i = 0; i < squarings; i++) {
        Mat tmpX = phiX.clone(), tmpY = phiY.clone();
        remap(tmpX, phiX, tmpX, tmpY, INTER_LINEAR);
        remap(tmpY, phiY, tmpX, tmpY, INTER_LINEAR);
    }
}

// --------------------------------------------------
// Dessin de la grille déformée par (phiX,phiY)
// --------------------------------------------------
void drawDiffeoGrid(const Mat& phiX, const Mat& phiY, Mat& output, int step) {
    output = Mat(phiX.size(), CV_8UC3, Scalar(255,255,255));
    int w = phiX.cols, h = phiX.rows;
    int cols = w/step, rows = h/step;

    vector<Point2f> P; P.reserve((cols+1)*(rows+1));
    for (int j = 0; j <= rows; j++) {
        for (int i = 0; i <= cols; i++) {
            float fx = i*step, fy = j*step;
            int ix = min(int(fx), w-2), iy = min(int(fy), h-2);
            float dx = fx - ix, dy = fy - iy;
            float X00 = phiX.at<float>(iy,ix),   X10 = phiX.at<float>(iy,ix+1);
            float X01 = phiX.at<float>(iy+1,ix), X11 = phiX.at<float>(iy+1,ix+1);
            float Y00 = phiY.at<float>(iy,ix),   Y10 = phiY.at<float>(iy,ix+1);
            float Y01 = phiY.at<float>(iy+1,ix), Y11 = phiY.at<float>(iy+1,ix+1);
            float wx = (1-dx)*(1-dy)*X00 + dx*(1-dy)*X10 + (1-dx)*dy*X01 + dx*dy*X11;
            float wy = (1-dx)*(1-dy)*Y00 + dx*(1-dy)*Y10 + (1-dx)*dy*Y01 + dx*dy*Y11;
            P.emplace_back(wx, wy);
        }
    }

    for (int j = 0; j <= rows; j++) {
        for (int i = 0; i <= cols; i++) {
            int idx = j*(cols+1) + i;
            Point2f p = P[idx];
            if (i < cols) {
                Point2f q = P[idx+1];
                line(output, p, q, Scalar(0,0,0), 1);
            }
            if (j < rows) {
                Point2f q = P[idx + (cols+1)];
                line(output, p, q, Scalar(0,0,0), 1);
            }
        }
    }
}

// --------------------------------------------------
// main(): traitement d'un dossier de PNG
// --------------------------------------------------
int main() {
    // Demande du dossier d'images
    cout << "Entrez le dossier d'images (ex: ./RubberWhale) : ";
    string folder;
    getline(cin, folder);

    // Récupère tous les PNG triés
    vector<String> files;
    glob(folder + "/*.png", files, false);
    if (files.empty()) {
        cerr << "Aucune image PNG trouvée dans " << folder << "\n";
        return -1;
    }

    // Lecture de la première image
    Mat frame_prev = imread(files[0], IMREAD_COLOR);
    if (frame_prev.empty()) {
        cerr << "Impossible de lire " << files[0] << "\n";
        return -1;
    }
    Mat gray_prev;
    cvtColor(frame_prev, gray_prev, COLOR_BGR2GRAY);
    gray_prev.convertTo(gray_prev, CV_32F, 1.0/255.0);
    gray_prev = downsample(gray_prev, 4);

    // (Optionnel) VideoWriter
    double fps = 30;
    int W = gray_prev.cols, H = gray_prev.rows;
    VideoWriter writer(
        "diffeo_output.avi",
        VideoWriter::fourcc('M','J','P','G'),
        fps,
        Size(W, H)
    );
    if (!writer.isOpened()) {
        cerr << "Impossible d'ouvrir VideoWriter\n";
        return -1;
    }

    // Paramètres HS + diffeo
    const float alpha     = 0.01f;
    const int   hs_iters  = 16;
    const int   squarings = 4;
    const int   grid_step = 20;
    const float amp       = 5.0f;

    // Boucle sur toutes les images
    for (size_t i = 1; i < files.size(); ++i) {
        Mat frame_cur = imread(files[i], IMREAD_COLOR);
        if (frame_cur.empty()) continue;

        Mat gray_cur;
        cvtColor(frame_cur, gray_cur, COLOR_BGR2GRAY);
        gray_cur.convertTo(gray_cur, CV_32F, 1.0/255.0);
        gray_cur = downsample(gray_cur, 4);

        Mat flow_u, flow_v;
        computeHornSchunck(gray_prev, gray_cur, flow_u, flow_v, alpha, hs_iters);

        Mat phiX, phiY;
        exponentiateFlow(flow_u, flow_v, phiX, phiY, squarings);
        amplifyDisplacement(phiX, phiY, amp);

        Mat diffeo_vis;
        drawDiffeoGrid(phiX, phiY, diffeo_vis, grid_step);

        imshow("Grille diffeomorphe", diffeo_vis);
        imshow("Origine", gray_cur);
        if (waitKey(30) == 27) break;

        writer.write(diffeo_vis);
        // ou pour sauvegarder en PNG:
        // imwrite(format("out_%04zu.png", i), diffeo_vis);

        gray_prev = gray_cur.clone();
    }

    writer.release();
    return 0;
}