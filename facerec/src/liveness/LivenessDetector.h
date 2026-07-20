/**
 * @file LivenessDetector.h
 * @brief Own-implementation liveness detection (blink + mouth movement) for tracked faces.
 */

#pragma once

#include <map>
#include <vector>

#include <opencv2/core.hpp>

#include "liveness/BlinkDetector.h"
#include "detection/FaceDetector.h"
#include "liveness/MouthMovementDetector.h"

/**
 * @brief Fraction of dark pixels inside one image region.
 * @param gray 8-bit single-channel image.
 * @param roi Region of interest in image pixels; clipped to the image.
 * @param referenceMean Brightness the dark threshold is relative to; when not
 *        positive, the region's own mean intensity is used instead.
 * @return Fraction of pixels darker than a threshold relative to the
 *         reference, in `[0, 1]`, or `-1` when the clipped region is too
 *         small to measure.
 *
 * Normalizing the darkness threshold by a brightness reference keeps the
 * measure stable across lighting and skin tones. This one primitive powers
 * both liveness signals: eye openness and mouth-cavity darkness.
 */
float darkFraction(const cv::Mat &gray, cv::Rect roi, float referenceMean = -1.0f);

/**
 * @brief Measure how open one eye is from its surrounding image region.
 * @param gray 8-bit single-channel image the detection coordinates refer to.
 * @param eyeCenter Eye-center landmark in image pixels (YuNet indices 0 or 1).
 * @param faceWidth Width of the detected face box, used to size the eye region.
 * @param faceMean Mean brightness of the face (see `faceReferenceMean`) used
 *        as the dark-threshold reference; when not positive, the eye region's
 *        own mean is used as a fallback.
 * @return Openness value in `[0, 1]`, or `-1` when the region is unusable.
 *
 * The measure is the dark-pixel fraction of a small rectangle centered on the
 * eye landmark. An open eye contains the dark iris/pupil cluster; a closed
 * eye is mostly eyelid skin, so the fraction drops. The dark threshold is
 * anchored to the face's overall brightness rather than the eye region's own
 * mean: self-normalization would raise the threshold exactly when the bright
 * eyelid replaces the dark iris, letting lashes and eye-socket shadow keep
 * the fraction high and flattening the blink dip below detectability (seen
 * on real footage, especially with glasses). This is our EAR analog: the
 * bundled OpenCV has no dense eyelid landmarks, so openness is measured
 * photometrically instead of geometrically.
 */
float eyeOpenness(const cv::Mat &gray, const glm::vec2 &eyeCenter, float faceWidth, float faceMean);

/**
 * @brief Mean brightness of the central patch of a detected face box.
 * @param gray 8-bit single-channel image the detection coordinates refer to.
 * @param face Detection providing the face box.
 * @return Mean intensity of the middle half of the box, or `-1` when the
 *         clipped patch is too small to measure.
 *
 * Serves as the lighting/skin-tone reference for `eyeOpenness`. The middle
 * half of the box is dominated by skin (cheeks, nose, forehead) and excludes
 * the background and hair along the box edges.
 */
float faceReferenceMean(const cv::Mat &gray, const FaceDetection &face);

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
     * @brief How long a proven-live face keeps its LIVE verdict without new
     *        activity, in seconds.
     *
     * Once a track has produced at least one blink or mouth movement it has
     * demonstrated liveness, so its verdict is held LIVE through longer quiet
     * stretches than the initial decision window allows — sparse detected
     * blinks, or the shallow-blink opening seconds after a looped video
     * restarts, would otherwise flip a real person to PHOTO?. A face that has
     * *never* shown any activity gets no such grace and still flips after
     * `kDecisionWindowSeconds`, because a photo never blinked at all.
     */
    static constexpr float kLiveStickySeconds = 20.0f;

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
        BlinkDetector blink;          //< Blink detector fed by this face's raw eye-openness signal.
        MouthMovementDetector mouth;  //< Mouth detector fed by this face's mouth-darkness signal.
        float firstSeen = 0.0f;       //< Timestamp when this track was first observed.
        float lastSeen = 0.0f;        //< Timestamp of the most recent observation.
        float lastActivityAt = -1.0f; //< Timestamp of the last blink or mouth movement, negative before the first.
        int framesObserved = 0;       //< Frames this track has actually been sampled on.
        int blinkCount = 0;           //< Total blinks observed on this track.
        int mouthMovementCount = 0;   //< Total mouth movements observed on this track.
    };

    std::map<int, TrackState> states; //< Liveness state per stable track ID.
};
