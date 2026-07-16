/**
 * @file MouthMovementDetector.h
 * @brief Own-implementation mouth-movement detector over a mouth-darkness signal.
 */

#pragma once

/**
 * @brief Detect mouth movements (yawns, talking) in a mouth-darkness signal.
 *
 * The mirror image of BlinkDetector: a blink is a transient *dip* in eye
 * openness, while a mouth movement is a transient *rise* in mouth-region
 * darkness — an opening mouth exposes the dark oral cavity. The baseline
 * tracks the resting closed-mouth level (falls quickly, rises slowly, and is
 * frozen while an opening is in progress so the event is judged against the
 * pre-opening level throughout). Hysteresis uses absolute margins because
 * the resting darkness is close to zero, where ratio thresholds are
 * meaningless. Static darkness never fires: a signal that is high from the
 * very first sample (a photo of an open mouth) establishes the baseline
 * instead, and an "opening" that outlives any real yawn is re-adopted as the
 * new resting level. Fully own code operating on one float per frame; it
 * knows nothing about images.
 */
class MouthMovementDetector
{
  public:
    /**
     * @brief Darkness above the baseline required to count the mouth as open.
     */
    static constexpr float kOpenDelta = 0.12f;

    /**
     * @brief Margin the signal must fall back within to count as closed again.
     */
    static constexpr float kCloseDelta = 0.06f;

    /**
     * @brief Shortest opening accepted as a movement, in frames.
     *
     * Filters single-frame measurement spikes.
     */
    static constexpr int kMinOpenFrames = 2;

    /**
     * @brief Longest opening still treated as a movement in progress, in frames.
     *
     * Roughly 5 s at 30 fps — longer than any yawn; a mouth "open" beyond
     * this is a static feature (beard shadow, an open mouth held still) and
     * becomes the new baseline without counting as a movement.
     */
    static constexpr int kMaxOpenFrames = 150;

    /**
     * @brief Samples consumed before movements are trusted.
     */
    static constexpr int kWarmupFrames = 8;

    /**
     * @brief Baseline adaptation rate while the signal is below the baseline.
     */
    static constexpr float kBaselineFallRate = 0.35f;

    /**
     * @brief Baseline adaptation rate while the signal is above the baseline.
     */
    static constexpr float kBaselineRiseRate = 0.03f;

    /**
     * @brief Feed one darkness sample for the next frame.
     * @param darkness Per-frame mouth darkness; negative samples are skipped.
     * @return `true` exactly once per completed movement (on the closing frame).
     */
    bool addSample(float darkness);

  private:
    float baseline = 0.0f; //< Adaptive estimate of the closed-mouth darkness level.
    int frames = 0;        //< Valid samples consumed so far (for warmup).
    int openFrames = 0;    //< Length of the opening in progress, 0 when closed.
};
