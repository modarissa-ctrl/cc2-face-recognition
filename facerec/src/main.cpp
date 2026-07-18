/**
 * @file main.cpp
 * @brief Desktop entry point and headless utility modes for `facerec`.
 */

#include "ofMain.h"
#include "ofApp.h"
#include "AppPaths.h"
#include "BlinkDetector.h"
#include "FaceDetector.h"
#include "FaceRecognizer.h"
#include "FaceTracker.h"
#include "LivenessDetector.h"
#include "MouthMovementDetector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

namespace
{

/**
 * @brief Set up the headless YuNet detector.
 * @param detector Reference to the FaceDetector instance.
 * @return `true` if the detector was set up successfully.
 */
bool setupHeadlessDetector(FaceDetector &detector)
{
    if (!detector.setup(ofToDataPath(AppPaths::kYunetModel)))
    {
        std::fprintf(stderr, "could not load the YuNet model — run scripts/bootstrap.py first\n");
        return false;
    }
    return true;
}

/**
 * @brief Set up the headless SFace recognizer.
 * @param recognizer Reference to the FaceRecognizer instance.
 * @return `true` if the recognizer was set up successfully.
 */
bool setupHeadlessRecognizer(FaceRecognizer &recognizer)
{
    if (!recognizer.setup(ofToDataPath(AppPaths::kSfaceModel)))
    {
        std::fprintf(stderr, "could not load the SFace model — run scripts/bootstrap.py first\n");
        return false;
    }
    return true;
}

/**
 * @brief Load an image in headless mode.
 * @param path Image path relative to `bin/data` or absolute if supported by OF.
 * @param pixels Reference to an ofPixels object to store the loaded image.
 * @return `true` if the image was loaded successfully.
 */
bool loadHeadlessImage(const std::string &path, ofPixels &pixels)
{
    if (!ofLoadImage(pixels, ofToDataPath(path)))
    {
        std::fprintf(stderr, "could not load image: %s\n", path.c_str());
        return false;
    }
    return true;
}

/**
 * @brief Exercise the face tracker on synthetic detections.
 * @param check Callback that records one named pass/fail result.
 *
 * Runs entirely on hand-made bounding boxes, so it needs neither model files
 * nor a display and verifies the pure association logic: stable IDs under
 * small motion, position-based (not index-based) matching, dropout
 * tolerance, and fresh IDs for unrelated faces.
 */
void selftestTracker(const std::function<void(bool, const std::string &)> &check)
{
    auto det = [](float x, float y) {
        FaceDetection d;
        d.box = ofRectangle(x, y, 100, 100);
        return d;
    };

    FaceTracker tracker;
    auto first = tracker.update({det(100, 100), det(400, 100)});
    check(first.size() == 2 && first[0] > 0 && first[1] > 0 && first[0] != first[1],
          "tracker assigns distinct ids to new faces");

    auto moved = tracker.update({det(110, 105), det(395, 100)});
    check(moved == first, "tracker keeps ids across small motion");

    auto swapped = tracker.update({det(395, 100), det(115, 105)});
    check(swapped.size() == 2 && swapped[0] == first[1] && swapped[1] == first[0],
          "tracker ids follow position, not detection order");

    // One face disappears for a few frames (within the grace period), then
    // returns close to where it was last seen.
    for (int i = 0; i < FaceTracker::kMaxMissedFrames / 3; i++)
    {
        tracker.update({det(395, 100)});
    }
    auto returned = tracker.update({det(130, 110), det(395, 100)});
    check(returned.size() == 2 && returned[0] == first[0], "tracker keeps an id through a short dropout");

    auto stranger = tracker.update({det(800, 500)});
    check(stranger.size() == 1 && stranger[0] != first[0] && stranger[0] != first[1],
          "tracker gives a distant new face a fresh id");

    // Isolated centroid pass: a 70 px jump on a 100 px box drops IoU below
    // kMinIou (overlap 30x100 -> IoU ~0.18) yet keeps the center within the
    // gate (distance 70 < 0.75 * 100), so only the second (proximity) pass can
    // preserve the id here.
    FaceTracker jumpTracker;
    int jumpId = jumpTracker.update({det(100, 100)})[0];
    auto jumped = jumpTracker.update({det(170, 100)});
    check(jumped.size() == 1 && jumped[0] == jumpId, "tracker centroid pass links a fast jump that breaks overlap");

    // Beyond the grace period the track is retired, so a face returning to the
    // exact same spot must be issued a brand-new id rather than the stale one.
    for (int i = 0; i < FaceTracker::kMaxMissedFrames + 1; i++)
    {
        jumpTracker.update({});
    }
    auto reappeared = jumpTracker.update({det(170, 100)});
    check(reappeared.size() == 1 && reappeared[0] != jumpId, "tracker retires an id after the grace period");
}

/**
 * @brief Exercise the liveness pipeline on synthetic input.
 * @param check Callback that records one named pass/fail result.
 *
 * Covers all layers without models or a display: the blink and
 * mouth-movement state machines on hand-made 1-D signals, the photometric
 * eye/mouth measures on drawn patches, and the full per-track live/photo
 * decision on rendered frames driven by a fake clock.
 */
void selftestLiveness(const std::function<void(bool, const std::string &)> &check)
{
    // --- blink state machine on synthetic openness signals ---
    auto countBlinks = [](const std::vector<float> &signal) {
        BlinkDetector blink;
        int count = 0;
        for (float openness : signal)
        {
            count += blink.addSample(openness) ? 1 : 0;
        }
        return count;
    };

    std::vector<float> steadyEye(60, 0.2f);
    check(countBlinks(steadyEye) == 0, "blink detector ignores a steadily open eye");

    auto oneDip = steadyEye;
    for (int i = 30; i < 33; i++)
    {
        oneDip[i] = 0.02f;
    }
    check(countBlinks(oneDip) == 1, "blink detector counts a short dip as one blink");

    auto twoDips = oneDip;
    for (int i = 45; i < 48; i++)
    {
        twoDips[i] = 0.02f;
    }
    check(countBlinks(twoDips) == 2, "blink detector counts separated dips individually");

    std::vector<float> longClosure(120, 0.2f);
    for (int i = 30; i < 30 + BlinkDetector::kMaxClosedFrames * 2; i++)
    {
        longClosure[i] = 0.02f;
    }
    check(countBlinks(longClosure) == 0, "blink detector rejects a closure longer than a blink");

    // Alternating dips that stay above the closing threshold must not fire.
    std::vector<float> noisyEye(60);
    for (size_t i = 0; i < noisyEye.size(); i++)
    {
        noisyEye[i] = (i % 2 == 0) ? 0.2f : 0.15f;
    }
    check(countBlinks(noisyEye) == 0, "blink detector tolerates mild signal noise");

    // From real footage (test/videos): the open level drifts, but a genuine
    // blink dips ~0.2 below it and recovers. Even a shallow early blink (open
    // ~0.70, dip to ~0.50 for two samples) must fire — the ratio detector
    // this replaced missed exactly these, because 0.50 is only ~0.7 of the
    // open level, no deeper (in ratio) than the worst photo noise.
    std::vector<float> shallowBlink(60, 0.70f);
    for (int i = 30; i < 32; i++)
    {
        shallowBlink[i] = 0.50f;
    }
    check(countBlinks(shallowBlink) == 1, "blink detector catches a shallow early-blink dip by prominence");

    // A deeper, wider blink (up to the duration cap) still counts exactly once.
    std::vector<float> deepBlink(60, 0.58f);
    for (int i = 30; i < 36; i++)
    {
        deepBlink[i] = 0.33f;
    }
    check(countBlinks(deepBlink) == 1, "blink detector counts a deep multi-sample blink once");

    // The photo's mid-clip jitter oscillates open/dip every frame and never
    // settles open for two consecutive samples. Each lone dip is only a
    // single closed frame and never re-arms, so nothing fires.
    std::vector<float> chatter(60, 0.66f);
    for (size_t i = 12; i < chatter.size(); i++)
    {
        chatter[i] = (i % 2 == 0) ? 0.66f : 0.46f;
    }
    check(countBlinks(chatter) == 0, "blink detector ignores oscillating chatter that never settles open");

    // The photo's end-of-clip dip is deep but sustained — it stays low far
    // longer than any blink, so it is abandoned without a count.
    std::vector<float> sustained(90, 0.64f);
    for (int i = 30; i < 60; i++)
    {
        sustained[i] = 0.44f;
    }
    check(countBlinks(sustained) == 0, "blink detector abandons a sustained dark excursion");

    // A closure that recovers only partway and holds there (a squint or a
    // stare) never returns near the open level, so it is abandoned, not
    // counted, however long it lasts.
    std::vector<float> partialReopen(90, 0.60f);
    for (int i = 30; i < 90; i++)
    {
        partialReopen[i] = 0.42f;
    }
    check(countBlinks(partialReopen) == 0, "blink detector never fires when the eye never returns near open");

    // --- mouth-movement state machine on synthetic darkness signals ---
    auto countMovements = [](const std::vector<float> &signal) {
        MouthMovementDetector mouth;
        int count = 0;
        for (float darkness : signal)
        {
            count += mouth.addSample(darkness) ? 1 : 0;
        }
        return count;
    };

    std::vector<float> steadyMouth(60, 0.05f);
    check(countMovements(steadyMouth) == 0, "mouth detector ignores a steadily closed mouth");

    auto oneYawn = steadyMouth;
    for (int i = 30; i < 40; i++)
    {
        oneYawn[i] = 0.35f;
    }
    check(countMovements(oneYawn) == 1, "mouth detector counts a bounded opening as one movement");

    auto twoYawns = oneYawn;
    for (int i = 48; i < 54; i++)
    {
        twoYawns[i] = 0.35f;
    }
    check(countMovements(twoYawns) == 2, "mouth detector counts separated openings individually");

    auto spike = steadyMouth;
    spike[30] = 0.35f;
    check(countMovements(spike) == 0, "mouth detector rejects a single-frame spike");

    std::vector<float> openFromStart(120, 0.35f);
    check(countMovements(openFromStart) == 0, "mouth detector never fires on a mouth dark from the start");

    std::vector<float> endlessOpening(260, 0.05f);
    for (int i = 30; i < 30 + MouthMovementDetector::kMaxOpenFrames + 50; i++)
    {
        endlessOpening[i] = 0.35f;
    }
    check(countMovements(endlessOpening) == 0, "mouth detector re-baselines an opening that outlives a yawn");

    // --- photometric measures on drawn patches ---
    auto makeFace = [](float x) {
        FaceDetection face;
        face.box = ofRectangle(x, 50, 100, 100);
        face.landmarks[0] = glm::vec2(x + 30, 90);  // right eye
        face.landmarks[1] = glm::vec2(x + 70, 90);  // left eye
        face.landmarks[2] = glm::vec2(x + 50, 110); // nose, unused by liveness
        face.landmarks[3] = glm::vec2(x + 35, 130); // right mouth corner
        face.landmarks[4] = glm::vec2(x + 65, 130); // left mouth corner
        return face;
    };

    cv::Mat openEye(200, 200, CV_8UC1, cv::Scalar(180));
    cv::circle(openEye, cv::Point(100, 100), 6, cv::Scalar(40), cv::FILLED);
    cv::Mat closedEye(200, 200, CV_8UC1, cv::Scalar(180));
    float skinMean = 180.0f;
    float open = eyeOpenness(openEye, glm::vec2(100, 100), 100.0f, skinMean);
    float shut = eyeOpenness(closedEye, glm::vec2(100, 100), 100.0f, skinMean);
    check(open > 0.1f && shut >= 0.0f && shut < 0.02f, "eye openness separates a dark iris from uniform eyelid skin");
    check(eyeOpenness(openEye, glm::vec2(-20, -20), 100.0f, skinMean) < 0.0f,
          "eye openness reports an off-frame eye region as unmeasurable");

    // The dark threshold is anchored to the face brightness, so a closed eye
    // sitting in a shadowed socket (darker than the face, but uniform) still
    // reads as far less open than one with an iris — the self-normalized
    // variant collapsed this contrast and made real blinks undetectable.
    cv::Mat shadowedClosed(200, 200, CV_8UC1, cv::Scalar(140));
    cv::Mat shadowedOpen = shadowedClosed.clone();
    cv::circle(shadowedOpen, cv::Point(100, 100), 6, cv::Scalar(40), cv::FILLED);
    float shadowShut = eyeOpenness(shadowedClosed, glm::vec2(100, 100), 100.0f, skinMean);
    float shadowOpen = eyeOpenness(shadowedOpen, glm::vec2(100, 100), 100.0f, skinMean);
    check(shadowOpen > shadowShut + 0.1f, "eye openness keeps blink contrast inside a shadowed eye socket");

    FaceDetection refFace = makeFace(50);
    cv::Mat uniformFace(200, 200, CV_8UC1, cv::Scalar(180));
    float refMean = faceReferenceMean(uniformFace, refFace);
    check(refMean > 175.0f && refMean < 185.0f, "face reference mean samples the central face patch");

    FaceDetection mouthFace = makeFace(50);
    cv::Mat openMouth(200, 200, CV_8UC1, cv::Scalar(180));
    cv::ellipse(openMouth, cv::Point(100, 138), cv::Size(12, 9), 0, 0, 360, cv::Scalar(40), cv::FILLED);
    cv::Mat closedMouth(200, 200, CV_8UC1, cv::Scalar(180));
    float agape = faceMouthDarkness(openMouth, mouthFace);
    float lipsTogether = faceMouthDarkness(closedMouth, mouthFace);
    check(agape > 0.2f && lipsTogether >= 0.0f && lipsTogether < 0.02f,
          "mouth darkness separates an open cavity from closed lips");

    // --- end-to-end live/photo verdicts on rendered frames, fake clock ---
    // Three synthetic 100 px faces on bright uniform "skin": open eyes are
    // dark discs (closed eyes simply skin), an open mouth is a dark ellipse.
    // Face 0 blinks periodically, face 1 yawns periodically with eyes always
    // open, face 2 (the "photo") never does either.
    std::vector<FaceDetection> faces = {makeFace(20), makeFace(180), makeFace(340)};
    std::vector<int> ids = {1, 2, 3};

    auto renderFrame = [&faces](bool blinkerEyesOpen, bool yawnerMouthOpen) {
        cv::Mat frame(200, 480, CV_8UC3, cv::Scalar(180, 180, 180));
        auto drawOpenEyes = [&frame](const FaceDetection &face) {
            for (int eye = 0; eye < 2; eye++)
            {
                cv::Point center(int(face.landmarks[eye].x), int(face.landmarks[eye].y));
                cv::circle(frame, center, 6, cv::Scalar(40, 40, 40), cv::FILLED);
            }
        };
        if (blinkerEyesOpen)
        {
            drawOpenEyes(faces[0]);
        }
        drawOpenEyes(faces[1]);
        drawOpenEyes(faces[2]);
        if (yawnerMouthOpen)
        {
            cv::Point mouthCenter(int(faces[1].box.x) + 50, 138);
            cv::ellipse(frame, mouthCenter, cv::Size(12, 9), 0, 0, 360, cv::Scalar(40, 40, 40), cv::FILLED);
        }
        return frame;
    };

    LivenessDetector liveness;
    constexpr float kFps = 30.0f;
    std::vector<LivenessDetector::Status> early;
    std::vector<LivenessDetector::Status> statuses;
    for (int frame = 0; frame < 360; frame++)
    {
        // Face 0 closes its eyes for 3 frames every 3 seconds; face 1 opens
        // its mouth for 20 frames every 4 seconds; 360 frames = 12 s of fake
        // time, comfortably past the 7 s decision window.
        int blinkPhase = frame % 90;
        bool blinkerOpen = blinkPhase < 60 || blinkPhase > 62;
        int yawnPhase = frame % 120;
        bool yawnerOpen = yawnPhase >= 60 && yawnPhase < 80;
        statuses = liveness.update(renderFrame(blinkerOpen, yawnerOpen), faces, ids, frame / kFps);
        if (frame == 45)
        {
            early = statuses;
        }
    }
    check(early.size() == 3 && early[0].verdict == LivenessDetector::Verdict::Pending &&
              early[1].verdict == LivenessDetector::Verdict::Pending &&
              early[2].verdict == LivenessDetector::Verdict::Pending,
          "liveness stays pending before the decision window elapses");
    check(statuses.size() == 3 && statuses[0].verdict == LivenessDetector::Verdict::Live &&
              statuses[0].blinkCount >= 2,
          "liveness marks a blinking face LIVE");
    check(statuses[1].verdict == LivenessDetector::Verdict::Live && statuses[1].mouthMovementCount >= 2 &&
              statuses[1].blinkCount == 0,
          "liveness marks a yawning face LIVE without any blink");
    check(statuses[2].verdict == LivenessDetector::Verdict::NoActivity && statuses[2].blinkCount == 0 &&
              statuses[2].mouthMovementCount == 0,
          "liveness flags a motionless face as a possible photo");

    // --- a LIVE verdict expires back to NoActivity after a still window ---
    // A single face blinks a few times in the first few seconds, then holds
    // perfectly still. It must read LIVE right after blinking and then fall
    // back to PHOTO? once a full decision window passes without any activity.
    FaceDetection loneFace = makeFace(20);
    std::vector<FaceDetection> lone = {loneFace};
    std::vector<int> loneIds = {1};
    auto renderLone = [&loneFace](bool eyesOpen) {
        cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(180, 180, 180));
        if (eyesOpen)
        {
            for (int eye = 0; eye < 2; eye++)
            {
                cv::Point center(int(loneFace.landmarks[eye].x), int(loneFace.landmarks[eye].y));
                cv::circle(frame, center, 6, cv::Scalar(40, 40, 40), cv::FILLED);
            }
        }
        return frame;
    };

    LivenessDetector expiring;
    LivenessDetector::Status whileActive;
    LivenessDetector::Status afterStill;
    // 780 frames = 26 s of fake time. A proven-live face holds LIVE for the
    // sticky window (20 s) after its last activity, so the run must extend
    // well past 20 s from the final blink to see the verdict expire.
    for (int frame = 0; frame < 780; frame++)
    {
        // Three 3-frame blinks in the first ~3 s (frames 27-29, 57-59, 87-89),
        // then eyes stay open for the remaining ~23 s — past the sticky window.
        bool eyesOpen = frame >= 90 || frame % 30 < 27;
        afterStill = expiring.update(renderLone(eyesOpen), lone, loneIds, frame / kFps)[0];
        if (frame == 120)
        {
            whileActive = afterStill;
        }
    }
    check(whileActive.verdict == LivenessDetector::Verdict::Live && whileActive.blinkCount >= 2,
          "liveness reads LIVE while a recent blink is inside the window");
    check(afterStill.verdict == LivenessDetector::Verdict::NoActivity,
          "liveness expires a proven-live verdict to PHOTO? after the sticky window");
}

/**
 * @brief Run dependency, model, and tracker self-checks without creating a window.
 * @return Process exit code compatible with CI usage.
 */
int runSelftest()
{
    bool allPassed = true;
    auto check = [&allPassed](bool ok, const std::string &what) {
        ofLogNotice("facerec") << (ok ? "[ok]   " : "[FAIL] ") << what;
        if (!ok)
            allPassed = false;
    };

    ofLogNotice("facerec") << "openFrameworks " << ofGetVersionInfo();
    ofLogNotice("facerec") << "OpenCV " << cv::getVersionString();

    int cvVersion = CV_VERSION_MAJOR * 10000 + CV_VERSION_MINOR * 100 + CV_VERSION_REVISION;
    check(cvVersion >= 40504, "OpenCV >= 4.5.4 (required by YuNet)");

    std::string yunetPath = ofToDataPath(AppPaths::kYunetModel);
    std::string sfacePath = ofToDataPath(AppPaths::kSfaceModel);
    check(ofFile::doesFileExist(yunetPath), "YuNet model file in data/models");
    check(ofFile::doesFileExist(sfacePath), "SFace model file in data/models");

    FaceDetector yunet;
    check(yunet.setup(yunetPath), "cv::FaceDetectorYN loads the YuNet model");
    try
    {
        auto recognizer = cv::FaceRecognizerSF::create(sfacePath, "");
        check(recognizer != nullptr, "cv::FaceRecognizerSF loads the SFace model");
    }
    catch (const cv::Exception &e)
    {
        check(false, std::string("cv::FaceRecognizerSF loads the SFace model — ") + e.what());
    }

    selftestTracker(check);
    selftestLiveness(check);

    ofLogNotice("facerec") << (allPassed ? "selftest passed" : "selftest FAILED");
    return allPassed ? 0 : 1;
}

/**
 * @brief Detect faces in one image and print the results.
 * @param path Image path relative to `bin/data` or absolute if supported by OF.
 * @return Process exit code.
 */
int runHeadlessDetect(const std::string &path)
{
    FaceDetector detector;
    if (!setupHeadlessDetector(detector))
    {
        return 1;
    }

    ofPixels pixels;
    if (!loadHeadlessImage(path, pixels))
    {
        return 1;
    }

    auto detections = detectInPixels(detector, pixels);
    std::printf("faces: %zu\n", detections.size());
    for (size_t i = 0; i < detections.size(); i++)
    {
        const auto &d = detections[i];
        std::printf("face %zu: x=%.0f y=%.0f w=%.0f h=%.0f confidence=%.2f\n", i, d.box.x, d.box.y, d.box.width,
                    d.box.height, d.confidence);
    }
    return 0;
}

/**
 * @brief Detect and recognize faces in one image against the default gallery.
 * @param path Image path relative to `bin/data` or absolute if supported by OF.
 * @return Process exit code.
 *
 * The printed `name` field respects the default match threshold, while `best`
 * and `score` always expose the closest gallery entry for diagnostics.
 */
int runHeadlessIdentify(const std::string &path)
{
    FaceDetector detector;
    if (!setupHeadlessDetector(detector))
    {
        return 1;
    }
    FaceRecognizer recognizer;
    if (!setupHeadlessRecognizer(recognizer))
    {
        return 1;
    }
    if (recognizer.loadGallery(ofToDataPath(AppPaths::kGalleryDir), detector) == 0)
    {
        std::fprintf(stderr, "no usable gallery at data/gallery — run scripts/bootstrap.py first\n");
        return 1;
    }

    ofPixels pixels;
    if (!loadHeadlessImage(path, pixels))
    {
        return 1;
    }

    cv::Mat bgr = toBgr(std::move(pixels));
    auto detections = detector.detect(bgr);
    auto matches = recognizer.identify(bgr, detections);
    std::printf("gallery: %d person(s)\n", recognizer.personCount());
    std::printf("faces: %zu\n", detections.size());
    for (size_t i = 0; i < detections.size(); i++)
    {
        const auto &d = detections[i];
        const auto &m = matches[i];
        bool recognized = m.score >= FaceRecognizer::kDefaultMatchThreshold;
        std::string score = m.score < 0 ? "n/a" : ofToString(m.score, 2);
        std::printf("face %zu: name=%s best=%s score=%s x=%.0f y=%.0f w=%.0f h=%.0f\n", i,
                    recognized ? m.name.c_str() : "unknown", m.name.empty() ? "-" : m.name.c_str(), score.c_str(),
                    d.box.x, d.box.y, d.box.width, d.box.height);
    }
    return 0;
}

/**
 * @brief Short human-readable name for a liveness verdict.
 * @param verdict Per-face verdict to describe.
 * @return `"LIVE"`, `"PHOTO?"`, or `"PENDING"`.
 */
const char *verdictName(LivenessDetector::Verdict verdict)
{
    switch (verdict)
    {
    case LivenessDetector::Verdict::Live:
        return "LIVE";
    case LivenessDetector::Verdict::NoActivity:
        return "PHOTO?";
    default:
        return "PENDING";
    }
}

/**
 * @brief Replay a video through the real liveness pipeline and dump signals.
 * @param videoPath Path to a video file decodable by OpenCV's VideoIO backend.
 * @param targetFps Detection sampling rate to emulate, in frames per second.
 * @return Process exit code.
 *
 * A headless, read-only diagnostic: it runs the exact detect -> track ->
 * liveness chain on real footage so blink/mouth behaviour can be measured
 * instead of guessed. The interactive app only detects on a subset of frames
 * (detection is slow), so this mirrors that by processing one frame every
 * `round(videoFps / targetFps)` frames; timestamps still use the original
 * playback clock (`originalFrameIndex / videoFps`) so the decision window
 * matches real time. Prints one line per sampled frame that has a face and a
 * final summary. No detection or liveness logic is altered here.
 */
int runHeadlessLivenessReplay(const std::string &videoPath, float targetFps)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::fprintf(stderr, "could not open video: %s\n", videoPath.c_str());
        return 1;
    }

    FaceDetector detector;
    if (!setupHeadlessDetector(detector))
    {
        return 1;
    }

    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0.0)
    {
        // Some containers report no rate; assume a typical webcam rate so the
        // clock and sampling stride stay sane.
        videoFps = 30.0;
    }
    int stride = std::max(1, int(std::lround(videoFps / targetFps)));
    std::printf("video: %s  %.2f fps, sampling every %d frame(s) (~%.1f fps target)\n", videoPath.c_str(), videoFps,
                stride, targetFps);
    std::printf("frame     t(s)  id    eye  mouth  verdict  blinks mouths\n");

    FaceTracker tracker;
    LivenessDetector liveness;

    int frameIndex = 0;
    int sampledFrames = 0;
    int framesWithFace = 0;
    int liveFrames = 0;
    int photoFrames = 0;
    int pendingFrames = 0;
    int totalBlinks = 0;
    int totalMouthMovements = 0;

    cv::Mat bgr;
    while (cap.read(bgr))
    {
        if (bgr.empty() || frameIndex % stride != 0)
        {
            frameIndex++;
            continue;
        }
        sampledFrames++;

        float nowSeconds = float(frameIndex) / float(videoFps);
        auto faces = detector.detect(bgr);
        auto trackIds = tracker.update(faces);
        auto statuses = liveness.update(bgr, faces, trackIds, nowSeconds);

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        if (!faces.empty())
        {
            framesWithFace++;
            // Classify the sampled frame by its first (typically only) face and
            // accumulate the running per-track blink/mouth totals.
            switch (statuses[0].verdict)
            {
            case LivenessDetector::Verdict::Live:
                liveFrames++;
                break;
            case LivenessDetector::Verdict::NoActivity:
                photoFrames++;
                break;
            default:
                pendingFrames++;
                break;
            }
        }

        for (size_t i = 0; i < faces.size(); i++)
        {
            float eye = faceEyeOpenness(gray, faces[i]);
            float mouth = faceMouthDarkness(gray, faces[i]);
            int id = i < trackIds.size() ? trackIds[i] : 0;
            std::printf("%5d  %6.2f  %3d  %5.3f  %5.3f  %-7s  %5d  %5d\n", frameIndex, nowSeconds, id, eye, mouth,
                        verdictName(statuses[i].verdict), statuses[i].blinkCount, statuses[i].mouthMovementCount);
            totalBlinks = std::max(totalBlinks, statuses[i].blinkCount);
            totalMouthMovements = std::max(totalMouthMovements, statuses[i].mouthMovementCount);
        }
        frameIndex++;
    }

    std::printf("\nsummary: %d sampled frame(s), %d with a face; blinks=%d mouth=%d; "
                "verdict frames LIVE=%d PHOTO?=%d PENDING=%d\n",
                sampledFrames, framesWithFace, totalBlinks, totalMouthMovements, liveFrames, photoFrames,
                pendingFrames);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++)
    {
        args.push_back(argv[i]);
    }

    // Execute CLI-only modes before touching any GL state so CI and remote
    // hosts can run diagnostics without a display server.
    if (std::find(args.begin(), args.end(), "--selftest") != args.end())
    {
        return runSelftest();
    }
    // Diagnostic video replay: --liveness-replay <video> [fps]. Kept separate
    // from the single-image modes below because it takes an optional sampling
    // rate after the required path.
    if (auto flagIt = std::find(args.begin(), args.end(), "--liveness-replay"); flagIt != args.end())
    {
        auto pathIt = std::next(flagIt);
        if (pathIt == args.end() || (!pathIt->empty() && (*pathIt)[0] == '-'))
        {
            std::fprintf(stderr, "usage: facerec --liveness-replay <video> [fps]\n");
            return 1;
        }
        float targetFps = 23.0f;
        if (auto fpsIt = std::next(pathIt); fpsIt != args.end() && !fpsIt->empty() && (*fpsIt)[0] != '-')
        {
            targetFps = std::stof(*fpsIt);
        }
        return runHeadlessLivenessReplay(*pathIt, targetFps);
    }
    for (auto [flag, run] : {std::pair{"--detect", runHeadlessDetect}, std::pair{"--identify", runHeadlessIdentify}})
    {
        auto flagIt = std::find(args.begin(), args.end(), flag);
        if (flagIt == args.end())
        {
            continue;
        }
        auto imageIt = std::next(flagIt);
        if (imageIt == args.end() || (!imageIt->empty() && (*imageIt)[0] == '-'))
        {
            std::fprintf(stderr, "usage: facerec %s <image>\n", flag);
            return 1;
        }
        return run(*imageIt);
    }

    ofGLWindowSettings settings;
    settings.setSize(1024, 768);
    settings.title = "facerec";

    auto window = ofCreateWindow(settings);
    auto app = std::make_shared<ofApp>();
    app->args = args;
    ofRunApp(window, app);
    return ofRunMainLoop();
}
