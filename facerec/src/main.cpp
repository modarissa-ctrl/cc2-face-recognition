/**
 * @file main.cpp
 * @brief Desktop entry point and headless utility modes for `facerec`.
 */

#include "ofMain.h"
#include "ofApp.h"
#include "AppPaths.h"
#include "FaceDetector.h"
#include "FaceRecognizer.h"
#include "FaceTracker.h"

#include <algorithm>
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
