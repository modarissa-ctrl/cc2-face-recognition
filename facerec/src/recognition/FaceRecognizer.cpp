/**
 * @file FaceRecognizer.cpp
 * @brief Implementation of SFace recognition and gallery loading.
 */

#include "recognition/FaceRecognizer.h"

#include <algorithm>

#include "ofMain.h"

bool FaceRecognizer::setup(const std::string &modelPath)
{
    try
    {
        recognizer = cv::FaceRecognizerSF::create(modelPath, "");
        ofLogNotice("FaceRecognizer") << "loaded SFace model from " << modelPath;
    }
    catch (const cv::Exception &e)
    {
        ofLogError("FaceRecognizer") << "failed to load " << modelPath << ": " << e.what();
        recognizer = nullptr;
    }
    return isLoaded();
}

cv::Mat FaceRecognizer::embed(const cv::Mat &bgr, const FaceDetection &face)
{
    // SFace consumes YuNet's raw row format for geometric alignment, so the
    // decoded detection has to be serialized back before feature extraction.
    cv::Mat row = faceDetectionToYunetRow(face);

    try
    {
        cv::Mat aligned, feature;
        recognizer->alignCrop(bgr, row, aligned);
        recognizer->feature(aligned, feature);
        // feature() returns a view of an internal buffer; clone to keep it
        return feature.clone();
    }
    catch (const cv::Exception &e)
    {
        ofLogError("FaceRecognizer") << "embedding failed: " << e.what();
        return {};
    }
}

int FaceRecognizer::loadGallery(const std::string &dirPath, FaceDetector &detector)
{
    entries.clear();
    numPersons = 0;
    if (!isLoaded())
    {
        ofLogWarning("FaceRecognizer") << "cannot load gallery before the recognizer model is ready";
        return 0;
    }

    ofDirectory root(dirPath);
    if (!root.exists())
    {
        ofLogWarning("FaceRecognizer") << "no gallery folder at " << dirPath;
        return 0;
    }
    root.listDir();
    ofLogNotice("FaceRecognizer") << "loading gallery from " << dirPath;

    for (const auto &personFile : root.getFiles())
    {
        if (!personFile.isDirectory())
        {
            continue;
        }
        std::string name = personFile.getBaseName();
        ofLogNotice("FaceRecognizer") << "processing gallery person " << name;

        ofDirectory photos(personFile.getAbsolutePath());
        for (const std::string &ext : {"jpg", "jpeg", "png", "bmp", "pgm"})
        {
            photos.allowExt(ext);
        }
        photos.listDir();

        int before = int(entries.size());
        for (const auto &photo : photos.getFiles())
        {
            std::string path = photo.getAbsolutePath();
            ofPixels pixels;
            if (!ofLoadImage(pixels, path))
            {
                ofLogWarning("FaceRecognizer") << "could not load " << path;
                continue;
            }
            cv::Mat bgr = toBgr(std::move(pixels));
            auto faces = detector.detect(bgr);
            if (faces.empty())
            {
                ofLogWarning("FaceRecognizer") << "no face found in " << path;
                continue;
            }
            // Gallery photos may contain multiple faces; keep the most
            // confident one so each image contributes exactly one identity
            // example.
            auto best =
                std::max_element(faces.begin(), faces.end(), [](const FaceDetection &a, const FaceDetection &b) {
                    return a.confidence < b.confidence;
                });
            cv::Mat feature = embed(bgr, *best);
            if (feature.empty())
            {
                continue;
            }
            entries.push_back({name, feature});
            ofLogNotice("FaceRecognizer") << name << ": embedded " << photo.getFileName();
        }
        if (int(entries.size()) > before)
        {
            numPersons++;
        }
        else
        {
            ofLogWarning("FaceRecognizer") << name << ": no usable photos, skipping";
        }
    }

    ofLogNotice("FaceRecognizer") << "gallery: " << entries.size() << " embedding(s) across " << numPersons
                                  << " person(s)";
    return int(entries.size());
}

std::vector<FaceMatch> FaceRecognizer::identify(const cv::Mat &bgr, const std::vector<FaceDetection> &faces)
{
    std::vector<FaceMatch> matches(faces.size());
    if (!isLoaded() || entries.empty())
    {
        return matches;
    }

    for (size_t i = 0; i < faces.size(); i++)
    {
        cv::Mat feature = embed(bgr, faces[i]);
        if (feature.empty())
        {
            continue;
        }
        for (const auto &entry : entries)
        {
            // Cosine similarity is monotonic for "best match", so keeping the
            // largest score is enough.
            float score = float(recognizer->match(feature, entry.feature, cv::FaceRecognizerSF::DisType::FR_COSINE));
            if (score > matches[i].score)
            {
                matches[i] = {entry.name, score};
            }
        }
    }
    return matches;
}
