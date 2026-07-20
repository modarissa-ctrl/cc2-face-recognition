/**
 * @file LivenessDetector.cpp
 * @brief Implementation of the liveness detector and its photometric signals.
 */

#include "liveness/LivenessDetector.h"

#include <algorithm>
#include <iterator>

#include <opencv2/imgproc.hpp>

#include "ofLog.h"

namespace
{

/**
 * @brief Eye-region half width as a fraction of the face-box width.
 *
 * A human eye spans roughly a fifth of the face width; the region is kept a
 * little wider so the iris stays inside it through small landmark jitter.
 */
constexpr float kEyeRoiHalfWidthFactor = 0.11f;

/**
 * @brief Eye-region half height as a fraction of the face-box width.
 *
 * Deliberately short so the (always dark) eyebrow stays outside the region —
 * otherwise a closed eye would still contain dark pixels and blinks would
 * barely change the measure.
 */
constexpr float kEyeRoiHalfHeightFactor = 0.07f;

/**
 * @brief Mouth-region half width as a fraction of the mouth-corner span.
 */
constexpr float kMouthRoiHalfWidthFactor = 0.6f;

/**
 * @brief Mouth-region half height as a fraction of the mouth-corner span.
 */
constexpr float kMouthRoiHalfHeightFactor = 0.45f;

/**
 * @brief Downward shift of the mouth region as a fraction of the corner span.
 *
 * The jaw drops when the mouth opens, so the dark cavity appears mostly
 * below the corner line.
 */
constexpr float kMouthRoiDownShiftFactor = 0.15f;

/**
 * @brief Smallest mouth-corner span accepted, as a fraction of the face width.
 *
 * Guards the region size against landmark jitter when the corners nearly
 * coincide (extreme head poses).
 */
constexpr float kMinMouthSpanFactor = 0.2f;

/**
 * @brief Pixels below this fraction of the reference brightness count as dark.
 */
constexpr float kDarkThresholdRatio = 0.7f;

/**
 * @brief Central fraction of the face box sampled for the brightness reference.
 *
 * The middle half of the box in both axes: dominated by skin, free of the
 * background and hair that touch the box edges.
 */
constexpr float kFaceRefPatchFraction = 0.5f;

/**
 * @brief Minimum usable region side length in pixels.
 */
constexpr int kMinRoiSide = 3;

} // namespace

float darkFraction(const cv::Mat &gray, cv::Rect roi, float referenceMean)
{
    roi &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (roi.width < kMinRoiSide || roi.height < kMinRoiSide)
    {
        return -1.0f;
    }
    cv::Mat patch = gray(roi);
    double reference = referenceMean > 0.0f ? referenceMean : cv::mean(patch)[0];
    if (reference <= 0.0)
    {
        // An all-black region carries no information (dead frame, total
        // shadow), so report it as unmeasurable rather than as a signal.
        return -1.0f;
    }
    int dark = cv::countNonZero(patch < kDarkThresholdRatio * reference);
    return float(dark) / float(patch.total());
}

float eyeOpenness(const cv::Mat &gray, const glm::vec2 &eyeCenter, float faceWidth, float faceMean)
{
    float halfW = kEyeRoiHalfWidthFactor * faceWidth;
    float halfH = kEyeRoiHalfHeightFactor * faceWidth;
    return darkFraction(
        gray, cv::Rect(int(eyeCenter.x - halfW), int(eyeCenter.y - halfH), int(2.0f * halfW), int(2.0f * halfH)),
        faceMean);
}

float faceReferenceMean(const cv::Mat &gray, const FaceDetection &face)
{
    float margin = (1.0f - kFaceRefPatchFraction) * 0.5f;
    cv::Rect patch(int(face.box.x + margin * face.box.width), int(face.box.y + margin * face.box.height),
                   int(kFaceRefPatchFraction * face.box.width), int(kFaceRefPatchFraction * face.box.height));
    patch &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (patch.width < kMinRoiSide || patch.height < kMinRoiSide)
    {
        return -1.0f;
    }
    return float(cv::mean(gray(patch))[0]);
}

float faceEyeOpenness(const cv::Mat &gray, const FaceDetection &face)
{
    // YuNet landmark order puts the two eyes at indices 0 and 1. Averaging the
    // measurable eyes halves the noise; a blink closes both eyes at once, so
    // the dip survives the averaging.
    float faceMean = faceReferenceMean(gray, face);
    float sum = 0.0f;
    int measured = 0;
    for (int eye = 0; eye < 2; eye++)
    {
        float openness = eyeOpenness(gray, face.landmarks[eye], face.box.width, faceMean);
        if (openness >= 0.0f)
        {
            sum += openness;
            measured++;
        }
    }
    return measured > 0 ? sum / float(measured) : -1.0f;
}

float faceMouthDarkness(const cv::Mat &gray, const FaceDetection &face)
{
    // YuNet landmark order puts the mouth corners at indices 3 and 4.
    glm::vec2 rightCorner = face.landmarks[3];
    glm::vec2 leftCorner = face.landmarks[4];
    glm::vec2 center = (rightCorner + leftCorner) * 0.5f;
    float span = std::max(glm::distance(rightCorner, leftCorner), kMinMouthSpanFactor * face.box.width);
    float halfW = kMouthRoiHalfWidthFactor * span;
    float halfH = kMouthRoiHalfHeightFactor * span;
    float centerY = center.y + kMouthRoiDownShiftFactor * span;
    return darkFraction(gray,
                        cv::Rect(int(center.x - halfW), int(centerY - halfH), int(2.0f * halfW), int(2.0f * halfH)));
}

std::vector<LivenessDetector::Status> LivenessDetector::update(const cv::Mat &bgr,
                                                               const std::vector<FaceDetection> &faces,
                                                               const std::vector<int> &trackIds, float nowSeconds)
{
    std::vector<Status> statuses(faces.size());
    if (!faces.empty() && !trackIds.empty())
    {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        for (size_t i = 0; i < faces.size() && i < trackIds.size(); i++)
        {
            auto [it, isNew] = states.try_emplace(trackIds[i]);
            TrackState &state = it->second;
            if (isNew)
            {
                state.firstSeen = nowSeconds;
            }
            state.lastSeen = nowSeconds;
            state.framesObserved++;

            // The blink detector locates blinks by prominence and handles the
            // noise itself, so the raw eye signal is fed straight in — a
            // moving average here would smear the sharp one-to-two-sample
            // blink troughs it relies on. The mouth signal is likewise raw.
            float openness = faceEyeOpenness(gray, faces[i]);
            float mouthDarkness = faceMouthDarkness(gray, faces[i]);
            // Raw per-frame signals for threshold tuning; visible only when
            // the log level is lowered to verbose.
            ofLogVerbose("liveness") << "track #" << trackIds[i] << " eye=" << openness << " mouth=" << mouthDarkness;
            if (state.blink.addSample(openness))
            {
                state.blinkCount++;
                state.lastActivityAt = nowSeconds;
                ofLogNotice("liveness") << "track #" << trackIds[i] << " blink #" << state.blinkCount
                                        << " (eye=" << openness << " t=" << nowSeconds << ")";
            }
            if (state.mouth.addSample(mouthDarkness))
            {
                state.mouthMovementCount++;
                state.lastActivityAt = nowSeconds;
                ofLogNotice("liveness") << "track #" << trackIds[i] << " mouth movement #" << state.mouthMovementCount
                                        << " (mouth=" << mouthDarkness << " t=" << nowSeconds << ")";
            }

            statuses[i].blinkCount = state.blinkCount;
            statuses[i].mouthMovementCount = state.mouthMovementCount;
            // A face that has already proven live keeps the verdict through a
            // longer quiet stretch than one still on probation, so sparse
            // detected blinks (or the shallow-blink opening of a looped video)
            // don't flip a real person to PHOTO?; a face that never showed any
            // activity gets only the initial window before it is flagged.
            bool provenLive = state.blinkCount + state.mouthMovementCount > 0;
            float activityGrace = provenLive ? kLiveStickySeconds : kDecisionWindowSeconds;
            if (state.lastActivityAt >= 0.0f && nowSeconds - state.lastActivityAt <= activityGrace)
            {
                statuses[i].verdict = Verdict::Live;
            }
            else if (nowSeconds - state.firstSeen >= kDecisionWindowSeconds &&
                     state.framesObserved >= kMinObservedFrames)
            {
                // Only flag a photo once the window has elapsed *and* the face
                // has actually been sampled on enough frames — the wall-clock
                // window alone would misfire after a pause or slow detection.
                statuses[i].verdict = Verdict::NoActivity;
            }
            // else: still Pending (default) — not enough observation yet.
        }
    }

    // Forget tracks the tracker itself has long dropped so their IDs can be
    // recycled by new faces without inheriting old activity history.
    for (auto it = states.begin(); it != states.end();)
    {
        it = (nowSeconds - it->second.lastSeen > kStaleTrackSeconds) ? states.erase(it) : std::next(it);
    }
    return statuses;
}

void LivenessDetector::reset()
{
    states.clear();
}
