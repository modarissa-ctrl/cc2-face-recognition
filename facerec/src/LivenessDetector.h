/**
 * @file LivenessDetector.h
 * @brief Own-implementation liveness detection (blink + mouth movement) for tracked faces.
 */

#pragma once

#include <map>
#include <vector>

#include <opencv2/core.hpp>

#include "BlinkDetector.h"
#include "FaceDetector.h"
#include "MouthMovementDetector.h"

/**
 * @brief Fraction of dark pixels inside one image region.
 * @param gray 8-bit single-channel image.
 * @param roi Region of interest in image pixels; clipped to the image.
 * @return Fraction of pixels darker than a threshold relative to the region's
 *         own mean intensity, in `[0, 1]`, or `-1` when the clipped region is
 *         too small to measure.
 *
 * Normalizing the darkness threshold by the region mean keeps the measure
 * stable across lighting and skin tones. This one primitive powers both
 * liveness signals: eye openness and mouth-cavity darkness.
 */
float darkFraction(const cv::Mat &gray, cv::Rect roi);

/**
 * @brief Measure how open one eye is from its surrounding image region.
 * @param gray 8-bit single-channel image the detection coordinates refer to.
 * @param eyeCenter Eye-center landmark in image pixels (YuNet indices 0 or 1).
 * @param faceWidth Width of the detected face box, used to size the eye region.
 * @return Openness value in `[0, 1]`, or `-1` when the region is unusable.
 *
 * The measure is the dark-pixel fraction of a small rectangle centered on the
 * eye landmark. An open eye contains the dark iris/pupil cluster; a closed
 * eye is nearly uniform eyelid skin, so the fraction collapses toward zero.
 * This is our EAR analog: the bundled OpenCV has no dense eyelid landmarks,
 * so openness is measured photometrically instead of geometrically.
 */
float eyeOpenness(const cv::Mat &gray, const glm::vec2 &eyeCenter, float faceWidth);

/**
 * @brief Combined eye openness for one detected face.
 * @param gray 8-bit single-channel image the detection coordinates refer to.
 * @param face Detection providing the two eye landmarks and the box size.
 * @return Mean openness of the measurable eyes, or `-1` when neither eye
 *         region is usable (face touching the frame border, tiny face).
 */
float faceEyeOpenness(const cv::Mat &gray, const FaceDetection &face);

/**
 * @brief Measure mouth-cavity darkness for one detected face.
 * @param gray 8-bit single-channel image the detection coordinates refer to.
 * @param face Detection providing the two mouth-corner landmarks (YuNet
 *        indices 3 and 4) and the box size.
 * @return Darkness value in `[0, 1]`, or `-1` when the region is unusable.
 *
 * The measure is the dark-pixel fraction of a rectangle spanning the mouth
 * corners and extending slightly downward, where the dark oral cavity
 * appears when the mouth opens (yawn, talking). Closed lips leave the region
 * mostly uniform skin, so the fraction stays low.
 */
float faceMouthDarkness(const cv::Mat &gray, const FaceDetection &face);

/**
 * @brief Blink- and mouth-movement-based live/photo classification for tracked faces.
 *
 * Each stable track ID (from FaceTracker) owns a BlinkDetector and a
 * MouthMovementDetector fed by the photometric signals above. A face that
 * blinked or moved its mouth within the decision window is labeled live; a
 * face observed for a full window without either is flagged as possibly a
 * photo, because a printed photo or a phone screen held to the camera does
 * neither. This is a demo of the technique, not anti-spoofing: it is
 * defeated by replayed videos, and misses people who stare motionlessly —
 * see the report/README limitations.
 */
class LivenessDetector
{
  public:
    /**
     * @brief Liveness classification for one tracked face.
     */
    enum class Verdict
    {
        Pending,   //< Not yet observed for a full decision window and no activity seen.
        Live,      //< Blinked or moved the mouth within the last decision window.
        NoActivity //< A full window without any blink or mouth movement — possibly a photo.
    };

    /**
     * @brief Per-face liveness result for the current frame.
     */
    struct Status
    {
        Verdict verdict = Verdict::Pending; //< Current live/photo classification.
        int blinkCount = 0;                 //< Total blinks seen on this track.
        int mouthMovementCount = 0;         //< Total mouth movements seen on this track.
    };

    /**
     * @brief Observation window for the live/photo decision, in seconds.
     *
     * People blink every 2–10 s on average; a face that produces no blink and
     * no mouth movement in this window is flagged, and a live verdict expires
     * after the same window without further activity.
     */
    static constexpr float kDecisionWindowSeconds = 7.0f;

    /**
     * @brief Seconds after which an unseen track's liveness state is discarded.
     *
     * Comfortably longer than the tracker's own dropout grace period so a
     * track that survives a brief detector miss keeps its activity history;
     * at low frame rates the tracker may still outlast this, in which case a
     * recovered track simply restarts its observation.
     */
    static constexpr float kStaleTrackSeconds = 2.0f;

    /**
     * @brief Minimum frames a track must be observed before it can be flagged.
     *
     * The decision window is measured in wall-clock seconds, but the signals
     * are only sampled on frames the detector actually processes. This gate
     * keeps the window from racing ahead of the frame stream — after a video
     * pause or under slow detection, wall-clock time can jump while almost no
     * frames arrive — so a face is never called a photo until it has truly
     * been watched for a while (and the state machines have cleared warmup).
     * At ~30 fps this is ~2 s, well inside the decision window.
     */
    static constexpr int kMinObservedFrames = 60;

    /**
     * @brief Advance blink and mouth state for one frame of tracked faces.
     * @param bgr 8-bit 3-channel BGR frame the detections refer to.
     * @param faces Detections from the current frame.
     * @param trackIds Stable track ID per detection, index-aligned with `faces`.
     * @param nowSeconds Monotonic timestamp of this frame in seconds.
     * @return Liveness status per detection, index-aligned with `faces`.
     */
    std::vector<Status> update(const cv::Mat &bgr, const std::vector<FaceDetection> &faces,
                               const std::vector<int> &trackIds, float nowSeconds);

    /**
     * @brief Drop all per-track liveness state.
     *
     * Must be called whenever the tracker resets, because track IDs restart
     * from 1 and stale state would otherwise be inherited by unrelated faces.
     */
    void reset();

  private:
    /**
     * @brief Liveness bookkeeping for one stable track ID.
     */
    struct TrackState
    {
        BlinkDetector blink;         //< Blink detector fed by this face's eye-openness signal.
        MouthMovementDetector mouth; //< Mouth detector fed by this face's mouth-darkness signal.
        float firstSeen = 0.0f;      //< Timestamp when this track was first observed.
        float lastSeen = 0.0f;       //< Timestamp of the most recent observation.
        float lastActivityAt = -1.0f; //< Timestamp of the last blink or mouth movement, negative before the first.
        int framesObserved = 0;      //< Frames this track has actually been sampled on.
        int blinkCount = 0;          //< Total blinks observed on this track.
        int mouthMovementCount = 0;  //< Total mouth movements observed on this track.
    };

    std::map<int, TrackState> states; //< Liveness state per stable track ID.
};
