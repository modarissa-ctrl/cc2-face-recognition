/**
 * @file BlinkDetector.h
 * @brief Own-implementation blink detector over a per-face eye-openness signal.
 */

#pragma once

/**
 * @brief Detect blinks in a per-face eye-openness signal.
 *
 * Keeps a slowly adapting baseline of the "eyes open" level (rises quickly,
 * decays slowly so a blink cannot drag it down) and applies hysteresis: the
 * signal must drop below `kCloseRatio * baseline` and then recover above
 * `kReopenRatio * baseline` within `kMaxClosedFrames` frames to count as one
 * blink. Longer closures (eyes shut, occlusion) are ignored. Fully own code
 * operating on one float per frame; it knows nothing about images.
 */
class BlinkDetector
{
  public:
    /**
     * @brief Fraction of the baseline below which the eye counts as closed.
     */
    static constexpr float kCloseRatio = 0.55f;

    /**
     * @brief Fraction of the baseline the signal must recover above to reopen.
     */
    static constexpr float kReopenRatio = 0.8f;

    /**
     * @brief Longest closure still accepted as a blink, in frames.
     *
     * Roughly 0.4 s at 30 fps; a real blink lasts 100–300 ms.
     */
    static constexpr int kMaxClosedFrames = 12;

    /**
     * @brief Minimum baseline level required before blinks are trusted.
     *
     * Below this the openness signal is too weak to distinguish a blink from
     * measurement noise (tiny or badly lit eye regions).
     */
    static constexpr float kMinBaseline = 0.02f;

    /**
     * @brief Samples consumed before the baseline is considered established.
     */
    static constexpr int kWarmupFrames = 8;

    /**
     * @brief Baseline adaptation rate while the signal is above the baseline.
     */
    static constexpr float kBaselineRiseRate = 0.35f;

    /**
     * @brief Baseline adaptation rate while the signal is below the baseline.
     */
    static constexpr float kBaselineDecayRate = 0.03f;

    /**
     * @brief Feed one openness sample for the next frame.
     * @param openness Per-frame eye openness; negative samples are skipped.
     * @return `true` exactly once per completed blink (on the reopen frame).
     */
    bool addSample(float openness);

  private:
    float baseline = 0.0f; //< Adaptive estimate of the eyes-open openness level.
    int frames = 0;        //< Valid samples consumed so far (for warmup).
    int closedFrames = 0;  //< Length of the closure in progress, 0 when open.
};
