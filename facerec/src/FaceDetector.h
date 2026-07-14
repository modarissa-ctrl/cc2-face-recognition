/**
 * @file FaceDetector.h
 * @brief YuNet-based face detection helpers and data structures.
 */

#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include "ofMain.h"

/**
 * @brief Number of facial landmarks produced by YuNet for each detection.
 */
inline constexpr int kNumLandmarks = 5;
/**
 * @brief Column index of the face-box left coordinate in a YuNet row.
 */
inline constexpr int kBoxXIndex = 0;
/**
 * @brief Column index of the face-box top coordinate in a YuNet row.
 */
inline constexpr int kBoxYIndex = 1;
/**
 * @brief Column index of the face-box width in a YuNet row.
 */
inline constexpr int kBoxWidthIndex = 2;
/**
 * @brief Column index of the face-box height in a YuNet row.
 */
inline constexpr int kBoxHeightIndex = 3;
/**
 * @brief Column index of the first landmark x coordinate in a YuNet row.
 */
inline constexpr int kLandmarkOffset = 4;
/**
 * @brief Total number of float columns in one YuNet detection row.
 *
 * The row layout is:
 * [x, y, w, h, (landmark x, landmark y) * 5, score]
 */
inline constexpr int kYunetRowSize = kLandmarkOffset + 2 * kNumLandmarks + 1;
/**
 * @brief Column index of the confidence score in a YuNet row.
 */
inline constexpr int kConfidenceIndex = kYunetRowSize - 1;

/**
 * @brief One detected face expressed in the original image coordinate space.
 */
struct FaceDetection
{
    /**
     * @brief Bounding rectangle of the detected face in pixels.
     */
    ofRectangle box;
    /**
     * @brief Facial landmarks in YuNet order.
     *
     * The order is right eye, left eye, nose tip, right mouth corner,
     * left mouth corner.
     */
    std::array<glm::vec2, kNumLandmarks> landmarks;
    /**
     * @brief Detector confidence in the range reported by YuNet.
     */
    float confidence = 0.0f;
};

/**
 * @brief Thin wrapper around OpenCV's YuNet detector.
 *
 * The class owns the OpenCV detector instance, configures its threshold, and
 * hides the resolution scaling that is applied to large images before
 * inference. Callers always receive boxes and landmarks mapped back to the
 * original input resolution.
 */
class FaceDetector
{
  public:
    /**
     * @brief Load the YuNet ONNX model.
     * @param modelPath Path to the YuNet model file.
     * @param scoreThreshold Minimum confidence kept by the detector.
     * @return `true` when the detector was created successfully.
     */
    bool setup(const std::string &modelPath, float scoreThreshold = 0.6f);

    /**
     * @brief Report whether the detector is ready for inference.
     * @return `true` when the YuNet instance exists.
     */
    bool isLoaded() const
    {
        return detector != nullptr;
    }

    /**
     * @brief Update the detector confidence threshold.
     * @param threshold New minimum confidence to keep.
     */
    void setScoreThreshold(float threshold);

    /**
     * @brief Detect faces in a BGR image.
     * @param bgr 8-bit 3-channel BGR image in OpenCV layout.
     * @return Detected faces mapped into the original image coordinates.
     */
    std::vector<FaceDetection> detect(const cv::Mat &bgr);

  private:
    cv::Ptr<cv::FaceDetectorYN> detector;
    /**
     * @brief Longest image side allowed before inference is downscaled.
     */
    static constexpr int maxInferenceSide = 1280;
};

/**
 * @brief Convert openFrameworks pixel data to an OpenCV BGR matrix.
 * @param pixels Source pixels copied by value because normalization mutates them.
 * @return 8-bit 3-channel BGR image ready for OpenCV processing.
 */
cv::Mat toBgr(ofPixels pixels);

/**
 * @brief Convert pixels to BGR and run detection in one call.
 * @param detector Detector instance to use.
 * @param pixels Source pixels to normalize and process.
 * @return Face detections for the provided image.
 */
std::vector<FaceDetection> detectInPixels(FaceDetector &detector, ofPixels pixels);

/**
 * @brief Encode a decoded detection back into YuNet's raw row layout.
 * @param face Face detection expressed in image pixels.
 * @return `1 x kYunetRowSize` `CV_32F` matrix compatible with
 *         `cv::FaceRecognizerSF::alignCrop()`.
 */
cv::Mat faceDetectionToYunetRow(const FaceDetection &face);
