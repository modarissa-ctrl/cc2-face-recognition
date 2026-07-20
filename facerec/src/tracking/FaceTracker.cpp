/**
 * @file FaceTracker.cpp
 * @brief Implementation of the IoU/centroid face tracker.
 */

#include "tracking/FaceTracker.h"

#include <algorithm>

float rectIoU(const ofRectangle &a, const ofRectangle &b)
{
    float interArea = a.getIntersection(b).getArea();
    if (interArea <= 0.0f)
    {
        return 0.0f;
    }
    float unionArea = a.getArea() + b.getArea() - interArea;
    return unionArea > 0.0f ? interArea / unionArea : 0.0f;
}

namespace
{

/**
 * @brief Centroid affinity between a track box and a detection box.
 * @param track Last known track rectangle.
 * @param detection Candidate detection rectangle.
 * @return Center distance normalized by the centroid gate, or a value >= 1
 *         when the detection lies outside the gate.
 *
 * Values below 1 are acceptable matches and smaller is better, which lets the
 * caller compare candidates of different box sizes with one number.
 */
float centroidCost(const ofRectangle &track, const ofRectangle &detection)
{
    float gate = FaceTracker::kCentroidGateFactor * std::max(track.width, track.height);
    if (gate <= 0.0f)
    {
        return 1.0f;
    }
    return glm::distance(glm::vec2(track.getCenter()), glm::vec2(detection.getCenter())) / gate;
}

} // namespace

std::vector<int> FaceTracker::update(const std::vector<FaceDetection> &faces)
{
    std::vector<int> ids(faces.size(), 0);
    std::vector<bool> trackMatched(tracks.size(), false);
    std::vector<bool> faceMatched(faces.size(), false);

    // Both passes are greedy best-pair-first: repeatedly link the globally
    // strongest remaining pair so one ambiguous detection cannot claim a
    // clearly better track from another detection.
    auto matchPass = [&](auto affinity, auto accept, bool preferLarger) {
        while (true)
        {
            float best = 0.0f;
            int bestTrack = -1;
            int bestFace = -1;
            for (size_t t = 0; t < tracks.size(); t++)
            {
                if (trackMatched[t])
                {
                    continue;
                }
                for (size_t f = 0; f < faces.size(); f++)
                {
                    if (faceMatched[f])
                    {
                        continue;
                    }
                    float value = affinity(tracks[t].box, faces[f].box);
                    if (!accept(value))
                    {
                        continue;
                    }
                    bool better = bestTrack < 0 || (preferLarger ? value > best : value < best);
                    if (better)
                    {
                        best = value;
                        bestTrack = int(t);
                        bestFace = int(f);
                    }
                }
            }
            if (bestTrack < 0)
            {
                break;
            }
            trackMatched[bestTrack] = true;
            faceMatched[bestFace] = true;
            tracks[bestTrack].box = faces[bestFace].box;
            tracks[bestTrack].missedFrames = 0;
            ids[bestFace] = tracks[bestTrack].id;
        }
    };

    // Pass 1: overlap. Consecutive video frames move faces only slightly, so
    // IoU is the strongest association signal.
    matchPass(rectIoU, [](float iou) { return iou >= kMinIou; }, true /* larger IoU is better */);
    // Pass 2: proximity. Catches fast motion and re-appearance after a short
    // dropout, where boxes no longer overlap but centers stay close.
    matchPass(centroidCost, [](float cost) { return cost < 1.0f; }, false /* smaller distance is better */);

    // Unmatched tracks age; dropping them only after a grace period keeps IDs
    // stable through brief detector misses.
    std::vector<Track> alive;
    alive.reserve(tracks.size() + faces.size());
    for (size_t t = 0; t < tracks.size(); t++)
    {
        Track track = tracks[t];
        if (!trackMatched[t] && ++track.missedFrames > kMaxMissedFrames)
        {
            continue;
        }
        alive.push_back(track);
    }
    tracks = std::move(alive);

    // Unmatched detections are new faces entering the scene.
    for (size_t f = 0; f < faces.size(); f++)
    {
        if (!faceMatched[f])
        {
            tracks.push_back({nextId, faces[f].box, 0});
            ids[f] = nextId;
            nextId++;
        }
    }
    return ids;
}

void FaceTracker::reset()
{
    tracks.clear();
    nextId = 1;
}
