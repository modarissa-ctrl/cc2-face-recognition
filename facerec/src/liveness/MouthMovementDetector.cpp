/**
 * @file MouthMovementDetector.cpp
 * @brief Implementation of the mouth-movement state machine.
 */

#include "liveness/MouthMovementDetector.h"

bool MouthMovementDetector::addSample(float darkness)
{
    if (darkness < 0.0f)
    {
        // Unmeasurable frame (face at the border, tiny face): hold the current
        // state instead of feeding garbage into the baseline.
        return false;
    }
    frames++;
    if (frames == 1)
    {
        // Adopt the first measurement as the resting level so a mouth that is
        // dark from the start (open-mouthed photo) never registers an opening.
        baseline = darkness;
        return false;
    }

    bool movement = false;
    if (openFrames > 0)
    {
        if (darkness < baseline + kCloseDelta)
        {
            // Closed again: a bounded opening is one movement; a single-frame
            // spike is measurement noise.
            movement = openFrames >= kMinOpenFrames;
            openFrames = 0;
        }
        else if (++openFrames > kMaxOpenFrames)
        {
            // The "opening" outlived any real yawn, so it is a static dark
            // feature (scarf, shadow, an open mouth held still): adopt it as
            // the new resting level without counting a movement.
            baseline = darkness;
            openFrames = 0;
        }
    }
    else if (frames > kWarmupFrames && darkness > baseline + kOpenDelta)
    {
        openFrames = 1;
    }

    if (openFrames == 0)
    {
        // Adapt only while the mouth is closed so an opening in progress is
        // judged against the pre-opening resting level for its whole
        // duration. Fall fast / rise slow keeps the baseline pinned to the
        // darker resting phase through lighting changes.
        float rate = darkness < baseline ? kBaselineFallRate : kBaselineRiseRate;
        baseline += (darkness - baseline) * rate;
    }
    return movement;
}
