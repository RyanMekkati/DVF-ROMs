#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

// Rappel pour savoir comment compiler le code
//g++ -std=c++11 -o Flux Flux.cpp `pkg-config --cflags --libs opencv4`
// ./Flux <Chemin.mp4> 

/************************************************** */
/******** Fonction de sous-échantillonnage  *********/
/************************************************** */

Mat downsample(const Mat& input, int factor) {
    Mat output(input.size(), input.type());
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            // Trouver le bloc correspondant
            int srcY = (y / factor) * factor;
            int srcX = (x / factor) * factor;

            // Définir la région du bloc
            int endY = min(srcY + factor, input.rows);
            int endX = min(srcX + factor, input.cols);

            // Calcul de la moyenne
            float sum = 0.0f;
            int count = 0;
            for (int i = srcY; i < endY; ++i) {
                for (int j = srcX; j < endX; ++j) {
                    sum += input.at<float>(i, j);
                    count++;
                }
            }
            float avg = sum / count;

            // Assigner la valeur de la moyenne au bloc
            for (int i = srcY; i < endY; ++i) {
                for (int j = srcX; j < endX; ++j) {
                    output.at<float>(i, j) = avg;
                }
            }
        }
    }
    return output;
}

/************************************************** */
/*********  Calcul du flux optique par H&S  *********/
/************************************************** */
void computeHornSchunck(const Mat& I1, const Mat& I2, Mat& u, Mat& v, float alpha, int max_iter) {

// Création des matrices dérivées et calcul des dérivées en x et y avec Sobel
    Mat Ix, Iy, It;
    Sobel(I1, Ix, CV_32F, 1, 0, 3); // Calcul de la dérivée partielle selon x (input,output,type,dx,dy,taille du noyau de Sobel)
    Sobel(I1, Iy, CV_32F, 0, 1, 3); // Calcul de la dérivée partielle selon y
    It = I2 - I1;                   // Calcul de la dérivée partielle selon le temps (car on prend t comme des unité de frame)

// On initie le champ de vecteur tous en pointant vers la droite de manière arbitraire (angle 0)
    u = Mat::ones(I1.size(), CV_32F) ;      // Composante horizontale initiale
    v = Mat::zeros(I1.size(), CV_32F);      // Composante verticale initiale

    for (int iter = 0; iter < max_iter; iter++) {
        Mat u_avg, v_avg;
        blur(u, u_avg, Size(3, 3)); // Convolution avec filtre moyenneur pour u
        blur(v, v_avg, Size(3, 3)); // Convolution avec filtre moyenneur pour v

// On itère sur les pixels pour trouver le nouvel estimé du flux par Gauss-Seidel
        for (int y = 0; y < I1.rows; y++) {
            for (int x = 0; x < I1.cols; x++) {
                float Ex = Ix.at<float>(y, x);
                float Ey = Iy.at<float>(y, x);
                float Et = It.at<float>(y, x);

                float numerator = Ex * u_avg.at<float>(y, x) +
                                Ey * v_avg.at<float>(y, x) + Et;
                float denominator = alpha * alpha + Ex * Ex + Ey * Ey;

                float update = numerator / denominator;

                u.at<float>(y, x) = u_avg.at<float>(y, x) - Ex * update;
                v.at<float>(y, x) = v_avg.at<float>(y, x) - Ey * update;
            }
        }
    }
}

/************************************************** */
/************ Construction de la grille *************/
/************************************************** */

void draw_sparse_flow(const Mat& u, const Mat& v, Mat& output, int step) {
    // Initiation de la grille avec la couleur (Bleu,Vert,Rouge)
    output = Mat(u.size(), CV_8UC3, Scalar(255, 255, 255));
    // Attribution des valeurs du flux à la grille
    for (int y = step / 2; y < u.rows; y += step) {
        for (int x = step / 2; x < u.cols; x += step) {
            float fx = u.at<float>(y, x);
            float fy = v.at<float>(y, x);

            // On prend une norme maximal pour les vecteurs de flux pour la lisibilité 
            float magnitude = sqrt(fx * fx + fy * fy);
            if (magnitude > 0.001) {
                fx /= magnitude; // Composante x unitaire
                fy /= magnitude; // Composante y unitaire
            }
            // Coordonnées du vecteur ainsi que la direction donnée par fx,fy
            // Point2f sont des points en 2D qui permet de former le vecteur dans OpenCV
            Point2f start(x,y);
            Point2f end(x + fx * 10, y + fy * 10); // Échelle uniforme

            // Dessin des vecteurs avec les couleurs (Bleu,Vert,Rouge)
            line(output, start, end, Scalar(0, 0, 0), 1); // Ligne représentant le vecteur (1 == 1 pixel d'épaisseur de ligne)
            circle(output, start, 1, Scalar(0, 0, 0), -1); // Point initial à l'origine (-1 == cercle plein)
        }
    }
}

/****************************************************/
/************ Partie principale du code *************/
/****************************************************/


int main(int argc, char* argv[]) {
    // Vérifie si un argument a été fourni
    if (argc < 2) {
        // Si aucun fichier n'a été donné, affiche le message d'utilisation
        cerr << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    // Récupère le chemin du fichier vidéo depuis les arguments en ligne de commande
    string video_file = argv[1];

    // Capture de la vidéo pour être lisible par l'algorithme
    VideoCapture cap(video_file);
    if (!cap.isOpened()) {
        cerr << "Erreur : Impossible d'ouvrir le fichier vidéo." << endl;
        return -1;
    }

    // Initialisation des matrices images importantes
    Mat frame_past, frame_past_grey, frame_present, frame_present_grey, flow_u, flow_v;

    // On convertit l'image en ton de gris pour le calcul du flux 
    cap >> frame_past;
    cvtColor(frame_past, frame_past_grey, COLOR_BGR2GRAY);

    // On applique une normalisation pour changer la plage des pixels de 0 à 1 (très utile pour la les calculs de flux optique)
    frame_past_grey.convertTo(frame_past_grey, CV_32F, 1.0 / 255.0);
    frame_past_grey = downsample(frame_past_grey, 4); // Sous-échantillonnage initial

    while (true) {
        cap >> frame_present;
        // Si on est arrivé à la fin de la vidéo (dernier frame)
        if (frame_present.empty()) break;

        // On convertit l'image en ton de gris pour le calcul du flux
        cvtColor(frame_present, frame_present_grey, COLOR_BGR2GRAY);
        frame_present_grey.convertTo(frame_present_grey, CV_32F, 1.0 / 255.0);
        frame_present_grey = downsample(frame_present_grey, 4); // Sous-échantillonnage

        // Rejet du background pour tout ce qui est plus petit qu'une valeur fixe (pour la visibilité)
        threshold(frame_present_grey, frame_present_grey, 0.5, 1.0, THRESH_TOZERO);

        // Calcul du flux optique
        computeHornSchunck(frame_past_grey, frame_present_grey, flow_u, flow_v, 0.01f, 16); 

        // On place les vecteurs sur la grille pour ce frame
        Mat sparse_flow;
        draw_sparse_flow(flow_u, flow_v, sparse_flow, 16); // Grille

        // Commande pour montrer les vidéos
        imshow("Flux optique simplifié", sparse_flow);
        imshow("Vidéo originale", frame_present_grey);

        // Le frame présent devient celui du passé pour la prochaine itération
        frame_past_grey = frame_present_grey.clone();

        // Commande pour sortir de la boucle en appuyant sur "esc"
        if (waitKey(100) == 27) break;
    }

    return 0;
}