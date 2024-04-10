#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void detectFaces(const cv::Mat& frame, cv::CascadeClassifier& faceCascade, std::vector<cv::Rect>& faces) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    faceCascade.detectMultiScale(gray, faces, 2, 3, 0, cv::Size(20, 20));

    for (const auto& face : faces) {
        cv::rectangle(frame, face, cv::Scalar(0, 0, 255), 2);
    }
}

void detectEyes(const cv::Mat& frame, std::vector<cv::Rect>& faces, cv::CascadeClassifier& eyesCascade) {
    for (auto& face : faces) {
        cv::Mat faceROI = frame(face);

        std::vector<cv::Rect> eyes;
        eyesCascade.detectMultiScale(faceROI, eyes, 3, 8, 0, cv::Size(5, 5));

        for (const auto& eye : eyes) {
            cv::Point center(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
            int radius = cvRound((eye.width + eye.height) * 0.25);
            cv::circle(frame, center, radius, cv::Scalar(200, 200, 0), 2);
        }
    }
}

void detectSmiles(const cv::Mat& frame, std::vector<cv::Rect>& faces, cv::CascadeClassifier& smileCascade) {
    for (auto& face : faces) {
        cv::Mat faceROI = frame(face);

        std::vector<cv::Rect> smiles;
        smileCascade.detectMultiScale(faceROI, smiles, 3, 35, 0, cv::Size(5, 5));

        for (const auto& smile : smiles) {
            cv::rectangle(frame, cv::Point(face.x + smile.x, face.y + smile.y),
                cv::Point(face.x + smile.x + smile.width, face.y + smile.y + smile.height),
                cv::Scalar(255, 0, 0), 2);
        }
    }
}

int main() {
    cv::CascadeClassifier faceCascade, eyesCascade, smileCascade;
    faceCascade.load("D:/haarcascades/haarcascade_frontalface_alt.xml");
    eyesCascade.load("D:/haarcascades/haarcascade_eye.xml");
    smileCascade.load("D:/haarcascades/haarcascade_smile.xml");

    cv::VideoCapture cap("D:/uwu.mp4");
    if (!cap.isOpened()) {
        std::cerr << "error" << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::vector<cv::Mat> processedFrames;

    auto start = std::chrono::steady_clock::now();

    omp_set_num_threads(3);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        std::vector<cv::Rect> faces;
        detectFaces(frame, faceCascade, faces);

#pragma omp parallel sections shared(frame, faces)
        {
#pragma omp section
            detectEyes(frame, faces, eyesCascade);

#pragma omp section
            detectSmiles(frame, faces, smileCascade);
        }

        processedFrames.push_back(frame.clone());
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "time: " << elapsed_seconds.count() << "s" << std::endl;

    cap.release();

    cv::VideoWriter video("D:/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight));

    for (const auto& frame : processedFrames) {
        video.write(frame);
        cv::imshow("UwU", frame);

        if (cv::waitKey(10) == 'q') {
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}
