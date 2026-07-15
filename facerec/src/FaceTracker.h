/**
 * @file FaceTracker.h
 * @brief Own-implementation IoU/centroid tracker that assigns stable face IDs.
 */

#pragma once

#include <vector>

#include "FaceDetector.h"

/**
 * @brief Compute the intersection-over-union of two rectangles.
 * @param a First rectangle.
 * @param b Second rectangle.
 * @return Overlap ratio in `[0, 1]`, where `0` means disjoint boxes.
 */
float rectIoU(const ofRectangle &a, const ofRectangle &b);

/**
 * @brief Frame-to-frame face tracker with stable integer IDs.
 *
 * The tracker associates detections between consecutive frames in two greedy
 * passes: first by strongest bounding-box overlap (IoU), then by nearest
 * centroid for pairs that overlap too little (fast motion, re-appearance
 * after a dropout). Tracks that stay unmatched survive for a short grace
 * period before they are dropped, so brief detector misses do not recycle
 * IDs. This is deliberately simple, model-free, and fully explainable own
 * work — no OpenCV tracking APIs involved.
 */
class FaceTracker
{
  public:
    /**
     * @brief Minimum IoU for the overlap pass to accept a track/detection pair.
     */
    static constexpr float kMinIou = 0.2f;

    /**
     * @brief Centroid gate as a fraction of the larger track box side.
     *
     * The centroid pass only links a detection whose center lies within
     * `kCentroidGateFactor * max(track width, track height)` of the track
     * center, so distant faces can never steal an existing ID.
     */
    static constexpr float kCentroidGateFactor = 0.75f;

    /**
     * @brief Consecutive unmatched frames tolerated before a track is dropped.
     *
     * Roughly half a second at 30 fps.
     */
    static constexpr int kMaxMissedFrames = 15;

    /**
     * @brief Associate one frame of detections with the tracked faces.
     * @param faces Detections from the current frame.
     * @return Stable track ID per detection, index-aligned with `faces`.
     *
     * Unmatched detections start new tracks; unmatched tracks age and are
     * dropped after `kMaxMissedFrames` consecutive misses.
     */
    std::vector<int> update(const std::vector<FaceDetection> &faces);

    /**
     * @brief Drop all tracks and restart ID numbering.
     */
    void reset();

  private:
    /**
     * @brief One face currently being tracked across frames.
     */
    struct Track
    {
        int id = 0;           //< Stable identifier shown to the user.
        ofRectangle box;      //< Last matched bounding box in image pixels.
        int missedFrames = 0; //< Consecutive frames without a matching detection.
    };

    std::vector<Track> tracks; //< Live tracks, including recently missed ones.
    int nextId = 1;            //< Identifier for the next newly created track.
};
