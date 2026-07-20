/**
 * @file HeadlessCommands.h
 * @brief Headless CLI utility modes extracted from `main.cpp`.
 */

#pragma once

#include "core/AppPaths.h"
#include "detection/FaceDetector.h"
#include "recognition/FaceRecognizer.h"
#include "tracking/FaceTracker.h"
#include "liveness/LivenessDetector.h"
#include "ofMain.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace headlessCommands
{
namespace detail
{

/**
 * @brief Set up the headless YuNet detector.
 * @param detector Reference to the FaceDetector instance.
 * @return `true` if the detector was set up successfully.
 */
inline bool setupHeadlessDetector(FaceDetector &detector)
{
    if (!detector.setup(ofToDataPath(AppPaths::kYunetModel)))
    {
        std::fprintf(stderr, "could not load the YuNet model -- run scripts/bootstrap.py first\n");
        return false;
    }
    return true;
}

/**
 * @brief Set up the headless SFace recognizer.
 * @param recognizer Reference to the FaceRecognizer instance.
 * @return `true` if the recognizer was set up successfully.
 */
inline bool setupHeadlessRecognizer(FaceRecognizer &recognizer)
{
    if (!recognizer.setup(ofToDataPath(AppPaths::kSfaceModel)))
    {
        std::fprintf(stderr, "could not load the SFace model -- run scripts/bootstrap.py first\n");
        return false;
    }
    return true;
}

/**
 * @brief Load an image in headless mode.
 * @param path Image path relative to `bin/data` or absolute if supported by OF.
 * @param pixels Reference to an ofPixels object to store the loaded image.
 * @return `true` if the image was loaded successfully.
 */
inline bool loadHeadlessImage(const std::string &path, ofPixels &pixels)
{
    if (!ofLoadImage(pixels, ofToDataPath(path)))
    {
        std::fprintf(stderr, "could not load image: %s\n", path.c_str());
        return false;
    }
    return true;
}

/**
 * @brief Short human-readable name for a liveness verdict.
 * @param verdict Per-face verdict to describe.
 * @return "LIVE", "PHOTO?", or "PENDING".
 */
inline const char *verdictName(LivenessDetector::Verdict verdict)
{
    switch (verdict)
    {
    case LivenessDetector::Verdict::Live:
        return "LIVE";
    case LivenessDetector::Verdict::NoActivity:
        return "PHOTO?";
    default:
        return "PENDING";
    }
}

} // namespace detail

/**
 * @brief Detect faces in one image and print the results.
 * @param path Image path relative to `bin/data` or absolute if supported by OF.
 * @return Process exit code.
 */
inline int runHeadlessDetect(const std::string &path)
{
    FaceDetector detector;
    if (!detail::setupHeadlessDetector(detector))
    {
        return 1;
    }

    ofPixels pixels;
    if (!detail::loadHeadlessImage(path, pixels))
    {
        return 1;
    }

    auto detections = detectInPixels(detector, pixels);
    std::printf("faces: %zu\n", detections.size());
    for (size_t i = 0; i < detections.size(); i++)
    {
        const auto &d = detections[i];
        std::printf("face %zu: x=%.0f y=%.0f w=%.0f h=%.0f confidence=%.2f\n", i, d.box.x, d.box.y, d.box.width,
                    d.box.height, d.confidence);
    }
    return 0;
}

/**
 * @brief Detect and recognize faces in one image against the default gallery.
 * @param path Image path relative to `bin/data` or absolute if supported by OF.
 * @return Process exit code.
 *
 * The printed `name` field respects the default match threshold, while `best`
 * and `score` always expose the closest gallery entry for diagnostics.
 */
inline int runHeadlessIdentify(const std::string &path)
{
    FaceDetector detector;
    if (!detail::setupHeadlessDetector(detector))
    {
        return 1;
    }
    FaceRecognizer recognizer;
    if (!detail::setupHeadlessRecognizer(recognizer))
    {
        return 1;
    }
    if (recognizer.loadGallery(ofToDataPath(AppPaths::kGalleryDir), detector) == 0)
    {
        std::fprintf(stderr, "no usable gallery at data/gallery -- run scripts/bootstrap.py first\n");
        return 1;
    }

    ofPixels pixels;
    if (!detail::loadHeadlessImage(path, pixels))
    {
        return 1;
    }

    cv::Mat bgr = toBgr(std::move(pixels));
    auto detections = detector.detect(bgr);
    auto matches = recognizer.identify(bgr, detections);
    std::printf("gallery: %d person(s)\n", recognizer.personCount());
    std::printf("faces: %zu\n", detections.size());
    for (size_t i = 0; i < detections.size(); i++)
    {
        const auto &d = detections[i];
        const auto &m = matches[i];
        bool recognized = m.score >= FaceRecognizer::kDefaultMatchThreshold;
        std::string score = m.score < 0 ? "n/a" : ofToString(m.score, 2);
        std::printf("face %zu: name=%s best=%s score=%s x=%.0f y=%.0f w=%.0f h=%.0f\n", i,
                    recognized ? m.name.c_str() : "unknown", m.name.empty() ? "-" : m.name.c_str(), score.c_str(),
                    d.box.x, d.box.y, d.box.width, d.box.height);
    }
    return 0;
}

/**
 * @brief Replay a video through the real liveness pipeline and dump signals.
 * @param videoPath Path to a video file decodable by OpenCV's VideoIO backend.
 * @param targetFps Detection sampling rate to emulate, in frames per second.
 * @return Process exit code.
 *
 * A headless, read-only diagnostic: it runs the exact detect -> track ->
 * liveness chain on real footage so blink/mouth behaviour can be measured
 * instead of guessed. The interactive app only detects on a subset of frames
 * (detection is slow), so this mirrors that by processing one frame every
 * `round(videoFps / targetFps)` frames; timestamps still use the original
 * playback clock (`originalFrameIndex / videoFps`) so the decision window
 * matches real time. Prints one line per sampled frame that has a face and a
 * final summary. No detection or liveness logic is altered here.
 */
inline int runHeadlessLivenessReplay(const std::string &videoPath, float targetFps)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::fprintf(stderr, "could not open video: %s\n", videoPath.c_str());
        return 1;
    }

    FaceDetector detector;
    if (!detail::setupHeadlessDetector(detector))
    {
        return 1;
    }

    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0.0)
    {
        // Some containers report no rate; assume a typical webcam rate so the
        // clock and sampling stride stay sane.
        videoFps = 30.0;
    }
    int stride = std::max(1, int(std::lround(videoFps / targetFps)));
    std::printf("video: %s  %.2f fps, sampling every %d frame(s) (~%.1f fps target)\n", videoPath.c_str(), videoFps,
                stride, targetFps);
    std::printf("frame     t(s)  id    eye  mouth  verdict  blinks mouths\n");

    FaceTracker tracker;
    LivenessDetector liveness;

    int frameIndex = 0;
    int sampledFrames = 0;
    int framesWithFace = 0;
    int liveFrames = 0;
    int photoFrames = 0;
    int pendingFrames = 0;
    int totalBlinks = 0;
    int totalMouthMovements = 0;

    cv::Mat bgr;
    while (cap.read(bgr))
    {
        if (bgr.empty() || frameIndex % stride != 0)
        {
            frameIndex++;
            continue;
        }
        sampledFrames++;

        float nowSeconds = float(frameIndex) / float(videoFps);
        auto faces = detector.detect(bgr);
        auto trackIds = tracker.update(faces);
        auto statuses = liveness.update(bgr, faces, trackIds, nowSeconds);

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        if (!faces.empty())
        {
            framesWithFace++;
            // Classify the sampled frame by its first (typically only) face and
            // accumulate the running per-track blink/mouth totals.
            switch (statuses[0].verdict)
            {
            case LivenessDetector::Verdict::Live:
                liveFrames++;
                break;
            case LivenessDetector::Verdict::NoActivity:
                photoFrames++;
                break;
            default:
                pendingFrames++;
                break;
            }
        }

        for (size_t i = 0; i < faces.size(); i++)
        {
            float eye = faceEyeOpenness(gray, faces[i]);
            float mouth = faceMouthDarkness(gray, faces[i]);
            int id = i < trackIds.size() ? trackIds[i] : 0;
            std::printf("%5d  %6.2f  %3d  %5.3f  %5.3f  %-7s  %5d  %5d\n", frameIndex, nowSeconds, id, eye, mouth,
                        detail::verdictName(statuses[i].verdict), statuses[i].blinkCount,
                        statuses[i].mouthMovementCount);
            totalBlinks = std::max(totalBlinks, statuses[i].blinkCount);
            totalMouthMovements = std::max(totalMouthMovements, statuses[i].mouthMovementCount);
        }
        frameIndex++;
    }

    std::printf("\nsummary: %d sampled frame(s), %d with a face; blinks=%d mouth=%d; "
                "verdict frames LIVE=%d PHOTO?=%d PENDING=%d\n",
                sampledFrames, framesWithFace, totalBlinks, totalMouthMovements, liveFrames, photoFrames,
                pendingFrames);
    return 0;
}

} // namespace headlessCommands
