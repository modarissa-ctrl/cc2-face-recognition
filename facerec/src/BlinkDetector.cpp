/**
 * @file BlinkDetector.cpp
 * @brief Implementation of the blink state machine.
 */

#include "BlinkDetector.h"

bool BlinkDetector::addSample(float openness)
{
    if (openness < 0.0f)
    {
        // Unmeasurable frame (face at the border, tiny face): hold the current
        // state instead of feeding garbage into the baseline.
        return false;
    }
    frames++;

    bool blink = false;
    bool established = frames > kWarmupFrames && baseline > kMinBaseline;
    if (closedFrames > 0)
    {
        if (openness > kReopenRatio * baseline)
        {
            // Reopened: only a short closure counts as a blink; anything
            // longer is eyes-shut or an occlusion, not a blink.
            blink = closedFrames <= kMaxClosedFrames;
            closedFrames = 0;
        }
        else
        {
            closedFrames++;
        }
    }
    else if (established && openness < kCloseRatio * baseline)
    {
        closedFrames = 1;
    }

    // Rise fast / decay slow: the baseline follows genuine lighting or
    // distance changes but is barely moved by a blink-length dip. Updated
    // after the state check so this frame is judged against the pre-existing
    // open level.
    float rate = openness > baseline ? kBaselineRiseRate : kBaselineDecayRate;
    baseline += (openness - baseline) * rate;
    return blink;
}
