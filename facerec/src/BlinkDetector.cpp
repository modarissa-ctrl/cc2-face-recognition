/**
 * @file BlinkDetector.cpp
 * @brief Implementation of the prominence-based blink state machine.
 */

#include "BlinkDetector.h"

bool BlinkDetector::addSample(float openness)
{
    if (openness < 0.0f)
    {
        // Unmeasurable frame (face at the border, tiny face): hold the current
        // state instead of feeding garbage into the reference.
        return false;
    }
    frames++;

    if (closedFrames == 0)
    {
        // --- open: decide whether a closure starts, then track the reference.
        // The entry test is evaluated first, against the open reference and
        // arming streak built up over the *preceding* frames, so this dipping
        // sample is judged against the pre-dip open level (and does not first
        // reset its own arming streak).
        bool established = frames > kWarmupFrames && openRef > kMinOpenRef;
        if (established && openStreak >= kArmStableFrames && openness < openRef - kDropMargin &&
            openness < kCloseRatio * openRef)
        {
            // Prominent drop from a stable-open plateau: a blink may be
            // starting. openRef is now frozen (closedFrames > 0) until the eye
            // reopens, so the whole closure is judged against the pre-dip level.
            closedFrames = 1;
            return false;
        }

        // Still open: extend the arming streak while near the open level (the
        // photo's chatter never holds near-open, so it never re-arms), and let
        // the peak-held reference jump up immediately but sink only slowly, so
        // a dip cannot erode the very prominence we measure against.
        openStreak = (openness >= openRef - kNearOpenMargin) ? openStreak + 1 : 0;
        openRef = openness > openRef ? openness : openRef + (openness - openRef) * kOpenRefDecay;
        return false;
    }

    // --- closed: wait for recovery, bounding how long a blink may last ---
    if (openness > openRef - kRecoverMargin)
    {
        // Reopened near the pre-dip level: a bounded closure is one blink. The
        // minimum length rejects lone single-frame spikes (the photo's
        // chatter), while a real blink always leaves at least a recovery-edge
        // sample below open; the maximum rejects deliberate closures.
        bool blink = closedFrames >= kMinClosedFrames && closedFrames <= kMaxClosedFrames;
        closedFrames = 0;
        openStreak = 0;
        return blink;
    }
    if (++closedFrames > kMaxClosedFrames)
    {
        // The low period outlived any real blink (the photo's sustained dark
        // excursion): abandon it without counting. Resetting the arming streak
        // forces a fresh stable-open plateau before another closure can start,
        // so a lingering excursion cannot repeatedly re-fire.
        closedFrames = 0;
        openStreak = 0;
    }
    return false;
}
