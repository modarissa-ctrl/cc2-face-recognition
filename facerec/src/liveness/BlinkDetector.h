/**
 * @file BlinkDetector.h
 * @brief Own-implementation blink detector over a per-face eye-openness signal.
 */

#pragma once

/**
 * @brief Detect blinks in a per-face eye-openness signal by prominence.
 *
 * A blink is a brief, sharp *dip* in eye openness: the dark iris/pupil is
 * momentarily hidden by the eyelid. This detector locates blinks by their
 * prominence relative to a peak-held estimate of the recent eyes-open level
 * (`openRef`), not by a fixed fraction of a slow baseline.
 *
 * The switch to prominence is deliberate and measured. On real footage the
 * open level drifts (≈0.72 near the camera start, ≈0.58 later), yet a blink
 * dips by roughly the same *absolute* amount (≈0.16–0.22) below whatever the
 * local open level is. A pure ratio test cannot exploit that: measured
 * blinks and the worst photo noise both sit around 0.65 of their local open
 * level, so no ratio separates them. Prominence from a fast-tracking
 * `openRef` normalizes the drift and, combined with a stable-open arming
 * gate and closure-duration bounds, rejects the two photo failure modes —
 * oscillating chatter (never settles open, so it never arms) and a sustained
 * dark excursion (stays low too long, so it is abandoned).
 *
 * Fully own code operating on one float per frame; it knows nothing about
 * images.
 */
class BlinkDetector
{
  public:
    /**
     * @brief Downward decay rate of the open reference while the eye is open.
     *
     * `openRef` jumps *up* to any higher openness immediately (it tracks the
     * open peak) but sinks only slowly toward lower open samples, so a single
     * dip cannot drag the reference down and swallow the very prominence the
     * detector measures. Slow enough that the reference still follows genuine
     * lighting or distance changes between blinks.
     */
    static constexpr float kOpenRefDecay = 0.05f;

    /**
     * @brief How close to `openRef` a sample must be to count as "open".
     *
     * Feeds the arming streak below. A real eye rests within a few hundredths
     * of its open peak; the photo's chattering signal bounces far more, so it
     * rarely produces consecutive near-open frames.
     */
    static constexpr float kNearOpenMargin = 0.08f;

    /**
     * @brief Consecutive near-open frames required before a new closure arms.
     *
     * The key defense against the photo's oscillating chatter: those dips
     * recover only to a level that keeps bouncing, never settling open for
     * two frames, so the detector never re-arms and cannot multi-count them.
     * A genuine blink is preceded by a stable-open plateau, so it arms
     * normally, and two real blinks close together are still separated by a
     * short open stretch.
     */
    static constexpr int kArmStableFrames = 2;

    /**
     * @brief Absolute drop below `openRef` required to enter the closed state.
     *
     * Measured genuine blinks drop ≈0.16–0.22 below the local open level;
     * the photo's shallow chatter dips only ≈0.13–0.15, so this margin keeps
     * most chatter from ever entering the closed state while still catching
     * the shallow early blinks a ratio test missed.
     */
    static constexpr float kDropMargin = 0.16f;

    /**
     * @brief Sample must also fall below this fraction of `openRef` to close.
     *
     * A belt-and-suspenders relative gate for bright scenes, where a fixed
     * absolute margin alone would be too easy to trip.
     */
    static constexpr float kCloseRatio = 0.80f;

    /**
     * @brief Recovery to within this of `openRef` reopens (ends the closure).
     */
    static constexpr float kRecoverMargin = 0.10f;

    /**
     * @brief Shortest closure still accepted as a blink, in frames.
     *
     * A lone single-frame dip that recovers immediately is measurement noise
     * — chiefly the photo's isolated chatter spikes. A genuine blink at the
     * ~20 fps the app samples always spans at least a trough plus a
     * recovery-edge sample, so requiring two closed frames rejects the spikes
     * without losing real blinks.
     */
    static constexpr int kMinClosedFrames = 2;

    /**
     * @brief Longest closure still accepted as a blink, in frames.
     *
     * A real blink spans up to ~6 samples at the ~20 fps the app actually
     * samples a 60 fps source; a longer low period is a deliberate closure or
     * the photo's sustained dark excursion and is abandoned without counting.
     */
    static constexpr int kMaxClosedFrames = 8;

    /**
     * @brief Samples consumed before the open reference is trusted.
     */
    static constexpr int kWarmupFrames = 8;

    /**
     * @brief Minimum open reference required before blinks are trusted.
     *
     * Below this the openness signal is too weak to distinguish a blink from
     * measurement noise (tiny or badly lit eye regions).
     */
    static constexpr float kMinOpenRef = 0.05f;

    /**
     * @brief Feed one openness sample for the next frame.
     * @param openness Per-frame eye openness; negative samples are skipped.
     * @return `true` exactly once per completed blink (on the reopen frame).
     */
    bool addSample(float openness);

  private:
    float openRef = 0.0f; //< Peak-held estimate of the recent eyes-open level.
    int frames = 0;       //< Valid samples consumed so far (for warmup).
    int closedFrames = 0; //< Length of the closure in progress, 0 when open.
    int openStreak = 0;   //< Consecutive recent near-open frames (arming gate).
};
