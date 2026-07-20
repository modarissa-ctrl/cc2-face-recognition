/**
 * @file FaceDetector.cpp
 * @brief Implementation of YuNet face detection helpers.
 */

#include "detection/FaceDetector.h"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "ofxCv.h"

namespace
{
const cv::Size kDefaultInputSize{320, 320}; //< Default input size for YuNet inference, in pixels.
}

bool FaceDetector::setup(const std::string &modelPath, float scoreThreshold)
{
    try
    {
        detector = cv::FaceDetectorYN::create(modelPath, "", kDefaultInputSize, scoreThreshold);
        ofLogNotice("FaceDetector") << "loaded YuNet model from " << modelPath << " with threshold "
                                    << ofToString(scoreThreshold, 2);
    }
    catch (const cv::Exception &e)
    {
        ofLogError("FaceDetector") << "failed to load " << modelPath << ": " << e.what();
        detector = nullptr;
    }
    return isLoaded();
}

void FaceDetector::setScoreThreshold(float threshold)
{
    if (detector)
    {
        detector->setScoreThreshold(threshold);
        ofLogNotice("FaceDetector") << "score threshold set to " << ofToString(threshold, 2);
    }
}

std::vector<FaceDetection> FaceDetector::detect(const cv::Mat &bgr)
{
    std::vector<FaceDetection> result;
    if (!detector || bgr.empty())
    {
        return result;
    }

    float scale = 1.0f;
    cv::Mat input = bgr;
    int longSide = std::max(bgr.cols, bgr.rows);
    if (longSide > maxInferenceSide)
    {
        scale = float(longSide) / maxInferenceSide;
        // Shrink oversized frames before inference, then scale boxes and
        // landmarks back so the rest of the pipeline stays resolution-agnostic.
        cv::resize(bgr, input, cv::Size(std::lround(bgr.cols / scale), std::lround(bgr.rows / scale)), 0, 0,
                   cv::INTER_AREA);
    }

    detector->setInputSize(input.size());
    // one row per face: x, y, w, h, five landmark (x, y) pairs, score
    cv::Mat faces;
    detector->detect(input, faces);

    for (int i = 0; i < faces.rows; i++)
    {
        const float *f = faces.ptr<float>(i);
        FaceDetection d;
        // YuNet emits all coordinates in a flat float row; decode them once
        // here into strongly typed app-facing structures.
        d.box = ofRectangle(f[kBoxXIndex] * scale, f[kBoxYIndex] * scale, f[kBoxWidthIndex] * scale,
                            f[kBoxHeightIndex] * scale);
        for (int k = 0; k < kNumLandmarks; k++)
        {
            d.landmarks[k] = {f[kLandmarkOffset + 2 * k] * scale, f[kLandmarkOffset + 2 * k + 1] * scale};
        }
        d.confidence = f[kConfidenceIndex];
        result.push_back(d);
    }
    return result;
}

cv::Mat toBgr(ofPixels pixels)
{
    // Normalize every input format to RGB first so OpenCV always receives a
    // predictable 8-bit BGR image, regardless of the source.
    pixels.setImageType(OF_IMAGE_COLOR);
    cv::Mat bgr;
    cv::cvtColor(ofxCv::toCv(pixels), bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

std::vector<FaceDetection> detectInPixels(FaceDetector &detector, ofPixels pixels)
{
    return detector.detect(toBgr(std::move(pixels)));
}

cv::Mat faceDetectionToYunetRow(const FaceDetection &face)
{
    // Rebuild the original YuNet row because OpenCV's SFace alignment API
    // expects that exact detector output layout rather than our decoded struct.
    cv::Mat row(1, kYunetRowSize, CV_32F);
    float *f = row.ptr<float>(0);
    f[kBoxXIndex] = face.box.x;
    f[kBoxYIndex] = face.box.y;
    f[kBoxWidthIndex] = face.box.width;
    f[kBoxHeightIndex] = face.box.height;
    for (int k = 0; k < kNumLandmarks; k++)
    {
        f[kLandmarkOffset + 2 * k] = face.landmarks[k].x;
        f[kLandmarkOffset + 2 * k + 1] = face.landmarks[k].y;
    }
    f[kConfidenceIndex] = face.confidence;
    return row;
}
