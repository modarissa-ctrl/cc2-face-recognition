/**
 * @file FaceRecognizer.h
 * @brief Gallery-backed SFace recognition interfaces.
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include "FaceDetector.h"

/**
 * @brief Best gallery match produced for one detected face.
 *
 * `name` stores the closest gallery person regardless of whether the caller
 * later accepts it as a recognized identity. The acceptance decision is made
 * outside this struct by comparing `score` against a threshold.
 */
struct FaceMatch
{
    /**
     * @brief Closest gallery identity, or empty when no embedding was produced.
     */
    std::string name;

    /**
     * @brief Cosine similarity score where larger values mean closer matches.
     */
    float score = -1.0f;
};

/**
 * @brief SFace recognizer with an in-memory gallery of labeled embeddings.
 *
 * Each face is aligned from YuNet landmarks, embedded into feature space, and
 * compared against all loaded gallery embeddings with cosine similarity.
 */
class FaceRecognizer
{
  public:
    /**
     * @brief Default cosine threshold recommended by the SFace model.
     */
    static constexpr float kDefaultMatchThreshold = 0.363f;

    /**
     * @brief Load the SFace ONNX model.
     * @param modelPath Path to the SFace model file.
     * @return `true` when the recognizer was created successfully.
     */
    bool setup(const std::string &modelPath);

    /**
     * @brief Report whether the recognizer is ready for embedding.
     * @return `true` when the recognizer instance exists.
     */
    bool isLoaded() const
    {
        return recognizer != nullptr;
    }

    /**
     * @brief Load a labeled face gallery from disk.
     * @param dirPath Folder containing one subfolder per person.
     * @param detector Detector used to locate faces in the gallery photos.
     * @return Number of successfully embedded gallery images.
     *
     * Each image contributes the single most confident detected face. Loading a
     * new gallery replaces any previously stored entries.
     */
    int loadGallery(const std::string &dirPath, FaceDetector &detector);

    /**
     * @brief Report whether any gallery embeddings are loaded.
     * @return `true` when the gallery contains at least one entry.
     */
    bool hasGallery() const
    {
        return !entries.empty();
    }

    /**
     * @brief Number of distinct people represented in the gallery.
     * @return Count of person folders with at least one usable embedding.
     */
    int personCount() const
    {
        return numPersons; //< Count of people with at least one usable embedding.
    }

    /**
     * @brief Match detected faces against the loaded gallery.
     * @param bgr Original image in BGR layout used to produce `faces`.
     * @param faces Face detections to embed and compare.
     * @return Best match for each input face, index-aligned with `faces`.
     */
    std::vector<FaceMatch> identify(const cv::Mat &bgr, const std::vector<FaceDetection> &faces);

  private:
    /**
     * @brief One labeled embedding stored in the gallery.
     */
    struct Entry
    {
        std::string name; //< Person identity label.
        cv::Mat feature;  //< 1x128 float embedding produced by SFace.
    };

    /**
     * @brief Align and embed one detected face.
     * @param bgr Source image in original BGR resolution.
     * @param face Detection describing the face to embed.
     * @return `1x128` float embedding, or an empty matrix on failure.
     */
    cv::Mat embed(const cv::Mat &bgr, const FaceDetection &face);

    cv::Ptr<cv::FaceRecognizerSF> recognizer; //< SFace model instance used for alignment and embedding.
    std::vector<Entry> entries;               //< Gallery entries containing labeled embeddings.
    int numPersons = 0;                       //< Number of distinct people in the gallery.
};
