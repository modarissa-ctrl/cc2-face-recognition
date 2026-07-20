/**
 * @file ofApp.cpp
 * @brief Interactive application flow for image, video, and webcam recognition.
 */

#include "ofApp.h"
#include "core/AppPaths.h"

#include <algorithm>
#include <utility>

namespace
{

/**
 * @brief Video file extensions accepted by the media loader.
 */
const std::vector<std::string> kVideoExtensions = {"mp4", "mov", "m4v", "avi", "mpg", "mpeg", "mkv", "webm"};

constexpr float kOverlayLineWidth = 2.0f;
constexpr float kLandmarkRadius = 3.0f;
constexpr float kOverlayLabelOffsetY = 6.0f;
constexpr int kOverlayRecognizedR = 80;
constexpr int kOverlayRecognizedG = 180;
constexpr int kOverlayRecognizedB = 255;
constexpr int kOverlayUnknownR = 80;
constexpr int kOverlayUnknownG = 255;
constexpr int kOverlayUnknownB = 120;

constexpr float kHudMarginLeft = 20.0f;
constexpr float kHudFacesBottomOffset = 44.0f;
constexpr float kHudInfoBottomOffset = 20.0f;
constexpr int kWebcamWidth = 1280;
constexpr int kWebcamHeight = 720;
constexpr int kWebcamFps = 30;

/**
 * @brief Decide whether a path should be treated as a video source.
 * @param path Candidate media path.
 * @return `true` when the extension belongs to the known video set.
 */
bool isVideoPath(const std::string &path)
{
    std::string ext = ofToLower(ofFilePath::getFileExt(path));
    return std::find(kVideoExtensions.begin(), kVideoExtensions.end(), ext) != kVideoExtensions.end();
}

} // namespace

void ofApp::setup()
{
    ofSetWindowTitle("facerec — M6 liveness");

    std::string modelPath = ofToDataPath(AppPaths::kYunetModel);
    bool loaded = detector.setup(modelPath, scoreThreshold);

    status = loaded ? "Open an image or video (O key, drag & drop, or the panel), or turn on the webcam."
                    : "FAILED to load the YuNet model — run scripts/bootstrap.py first.";

    if (!recognizer.setup(ofToDataPath(AppPaths::kSfaceModel)))
    {
        ofLogError("facerec") << "SFace model missing — recognition disabled, "
                                 "run scripts/bootstrap.py first";
    }
    else
    {
        recognizer.loadGallery(ofToDataPath(AppPaths::kGalleryDir), detector);
    }

    gui.setup("facerec");
    gui.add(openImageButton.setup("open image..."));
    gui.add(openVideoButton.setup("open video..."));
    gui.add(webcamOn);
    gui.add(webcamDeviceIndex);
    gui.add(refreshWebcamsButton.setup("refresh webcams"));
    gui.add(trackingOn);
    gui.add(livenessOn);
    gui.add(scoreThreshold);
    gui.add(matchThreshold);
    gui.add(loadGalleryButton.setup("load gallery..."));
    openImageButton.addListener(this, &ofApp::onOpenImage);
    openVideoButton.addListener(this, &ofApp::onOpenVideo);
    loadGalleryButton.addListener(this, &ofApp::onLoadGallery);
    refreshWebcamsButton.addListener(this, &ofApp::onRefreshWebcams);
    webcamOn.addListener(this, &ofApp::onWebcamToggle);
    webcamDeviceIndex.addListener(this, &ofApp::onWebcamDeviceIndexChanged);
    scoreThreshold.addListener(this, &ofApp::onScoreThreshold);
    trackingOn.addListener(this, &ofApp::onTrackingToggle);
    livenessOn.addListener(this, &ofApp::onLivenessToggle);

    refreshWebcamDevices(false);

    // a plain (non-flag) argument is an image or video to open at startup
    for (const auto &arg : args)
    {
        if (!arg.empty() && arg[0] != '-')
        {
            ofLogNotice("facerec") << "opening startup path " << arg;
            openPath(ofToDataPath(arg));
            break;
        }
    }
}

bool ofApp::openPath(const std::string &path)
{
    ofLogNotice("facerec") << "opening path " << path;
    return isVideoPath(path) ? loadVideo(path) : loadImage(path);
}

void ofApp::stopCurrentSource(bool keepWebcamToggle)
{
    if (mode != InputMode::None)
    {
        ofLogNotice("facerec") << "stopping source " << sourceName;
    }
    if (mode == InputMode::Video)
    {
        video.close();
    }
    else if (mode == InputMode::Webcam)
    {
        grabber.close();
        if (!keepWebcamToggle)
        {
            webcamOn.setWithoutEventNotifications(false);
        }
    }
    mode = InputMode::None;
    faces.clear();
    matches.clear();
    trackIds.clear();
    tracker.reset();
    liveStatus.clear();
    liveness.reset();
    detectMillis = 0.0f;
}

void ofApp::refreshWebcamDevices(bool keepSelection)
{
    int preferredDeviceId = -1;
    if (keepSelection && !webcamDevices.empty())
    {
        int currentIndex =
            ofClamp(static_cast<int>(webcamDeviceIndex.get()), 0, static_cast<int>(webcamDevices.size()) - 1);
        preferredDeviceId = webcamDevices[currentIndex].id;
    }

    webcamDevices = grabber.listDevices();

    int selectedIndex = 0;
    if (preferredDeviceId >= 0)
    {
        auto it =
            std::find_if(webcamDevices.begin(), webcamDevices.end(),
                         [preferredDeviceId](const ofVideoDevice &device) { return device.id == preferredDeviceId; });
        if (it != webcamDevices.end())
        {
            selectedIndex = static_cast<int>(std::distance(webcamDevices.begin(), it));
        }
    }

    webcamDeviceIndex.removeListener(this, &ofApp::onWebcamDeviceIndexChanged);
    if (webcamDevices.empty())
    {
        webcamDeviceIndex.set("webcam device", 0, 0, 0);
    }
    else
    {
        webcamDeviceIndex.set("webcam device", selectedIndex, 0, static_cast<int>(webcamDevices.size()) - 1);
    }
    updateWebcamDeviceControlLabel();
    webcamDeviceIndex.addListener(this, &ofApp::onWebcamDeviceIndexChanged);

    ofLogNotice("facerec") << "found " << webcamDevices.size() << " webcam device(s)";
    for (const auto &device : webcamDevices)
    {
        ofLogNotice("facerec") << "  [" << device.id << "] " << device.deviceName
                               << (device.bAvailable ? "" : " (unavailable)");
    }
}

void ofApp::updateWebcamDeviceControlLabel()
{
    if (webcamDevices.empty())
    {
        webcamDeviceIndex.setName("webcam device (none)");
        return;
    }

    int index = ofClamp(static_cast<int>(webcamDeviceIndex.get()), 0, static_cast<int>(webcamDevices.size()) - 1);
    const ofVideoDevice &device = webcamDevices[index];

    webcamDeviceIndex.setName("webcam " + ofToString(index + 1) + "/" + ofToString(webcamDevices.size()) + ": " +
                              device.deviceName);
}

bool ofApp::startWebcam()
{
    if (webcamDevices.empty())
    {
        refreshWebcamDevices(false);
    }
    if (webcamDevices.empty())
    {
        status = "No webcam devices found.";
        ofLogError("facerec") << status;
        webcamOn.setWithoutEventNotifications(false);
        stopCurrentSource();
        return false;
    }

    int index = ofClamp(static_cast<int>(webcamDeviceIndex.get()), 0, static_cast<int>(webcamDevices.size()) - 1);
    const ofVideoDevice &device = webcamDevices[index];

    stopCurrentSource(true);
    grabber.setDeviceID(device.id);
    grabber.setDesiredFrameRate(kWebcamFps);
    grabber.setup(kWebcamWidth, kWebcamHeight);
    if (!grabber.isInitialized())
    {
        status = "Could not open webcam: " + device.deviceName;
        ofLogError("facerec") << status;
        webcamOn.setWithoutEventNotifications(false);
        return false;
    }

    mode = InputMode::Webcam;
    sourceName = "webcam: " + device.deviceName;
    status.clear();
    lastLogMillis = 0;
    webcamOn.setWithoutEventNotifications(true);
    ofLogNotice("facerec") << "webcam started at " << kWebcamWidth << "x" << kWebcamHeight << " on [" << device.id
                           << "] " << device.deviceName;
    return true;
}

bool ofApp::loadImage(const std::string &path)
{
    ofImage next;
    if (!next.load(path))
    {
        status = "Could not load image: " + path;
        ofLogError("facerec") << status;
        return false;
    }
    stopCurrentSource();
    image = next;
    mode = InputMode::Image;
    sourceName = ofFilePath::getFileName(path);

    detectFrame(image.getPixels());
    ofLogNotice("facerec") << sourceName << ": " << faces.size() << " face(s) in " << detectMillis << " ms";
    return true;
}

bool ofApp::loadVideo(const std::string &path)
{
    ofVideoPlayer next;
    if (!next.load(path))
    {
        status = "Could not load video: " + path;
        ofLogError("facerec") << status;
        return false;
    }
    stopCurrentSource();
    video = std::move(next);
    video.setLoopState(OF_LOOP_NORMAL);
    video.play();
    mode = InputMode::Video;
    sourceName = ofFilePath::getFileName(path);
    status.clear();
    ofLogNotice("facerec") << "video source ready: " << sourceName;
    return true;
}

void ofApp::update()
{
    if (mode == InputMode::Video)
    {
        video.update();
        if (video.isFrameNew())
        {
            detectFrame(video.getPixels());
        }
    }
    else if (mode == InputMode::Webcam)
    {
        grabber.update();
        if (grabber.isFrameNew())
        {
            detectFrame(grabber.getPixels());
        }
    }

    // once-a-second heartbeat so video/webcam runs are observable in the log
    if (mode == InputMode::Video || mode == InputMode::Webcam)
    {
        uint64_t now = ofGetElapsedTimeMillis();
        if (now - lastLogMillis >= 1000)
        {
            lastLogMillis = now;
            std::string ids;
            for (size_t i = 0; i < trackIds.size(); i++)
            {
                ids += " #" + ofToString(trackIds[i]);
                if (i < liveStatus.size())
                {
                    // Verdict and blink/mouth counts per track keep video and
                    // webcam runs observable in the console, mirroring the
                    // on-screen flags.
                    if (liveStatus[i].verdict == LivenessDetector::Verdict::Live)
                    {
                        ids += ":LIVE(" + ofToString(liveStatus[i].blinkCount) + "b/" +
                               ofToString(liveStatus[i].mouthMovementCount) + "m)";
                    }
                    else if (liveStatus[i].verdict == LivenessDetector::Verdict::NoActivity)
                    {
                        ids += ":PHOTO?";
                    }
                }
            }
            ofLogNotice("facerec") << sourceName << ": " << faces.size() << " face(s)" << ids << ", "
                                   << ofToString(detectMillis, 1) << " ms/frame, " << ofToString(ofGetFrameRate(), 0)
                                   << " fps";
        }
    }
}

void ofApp::refreshStillImage()
{
    if (hasStillImage())
    {
        // Still images do not produce future frames, so threshold or gallery
        // changes require an explicit refresh to keep overlays in sync.
        detectMillis = 0.0f;
        detectFrame(image.getPixels());
    }
}

void ofApp::detectFrame(const ofPixels &pixels)
{
    uint64_t start = ofGetElapsedTimeMicros();
    cv::Mat bgr = toBgr(pixels);
    faces = detector.detect(bgr);
    matches = recognizer.hasGallery() ? recognizer.identify(bgr, faces) : std::vector<FaceMatch>();
    // Tracking only makes sense for temporal sources; a still image is a
    // single frame, so IDs would be meaningless there.
    if (trackingOn && (mode == InputMode::Video || mode == InputMode::Webcam))
    {
        trackIds = tracker.update(faces);
    }
    else
    {
        // The tracker is already reset on every transition into a non-tracking
        // state (source change in stopCurrentSource, toggle in
        // onTrackingToggle), so no stale track state can survive here — just
        // drop the now-meaningless labels for this frame.
        trackIds.clear();
    }
    // Liveness rides on the tracker: blink history only means something when
    // it can be pinned to a stable face ID, so it is active exactly when
    // tracking produced IDs for this frame (temporal source + tracking on).
    if (livenessOn && !trackIds.empty())
    {
        liveStatus = liveness.update(bgr, faces, trackIds, ofGetElapsedTimef());
    }
    else
    {
        liveStatus.clear();
    }
    float millis = (ofGetElapsedTimeMicros() - start) / 1000.0f;
    // Smooth only the UI-facing latency number so the overlay stays readable
    // while the underlying detections remain frame-accurate.
    detectMillis = detectMillis == 0.0f ? millis : ofLerp(detectMillis, millis, 0.15f);
}

std::pair<float, float> ofApp::sourceSize() const
{
    if (hasStillImage())
    {
        return {image.getWidth(), image.getHeight()};
    }
    else if (mode == InputMode::Video && video.isLoaded())
    {
        return {video.getWidth(), video.getHeight()};
    }
    else if (mode == InputMode::Webcam && grabber.isInitialized())
    {
        return {grabber.getWidth(), grabber.getHeight()};
    }
    return {0, 0};
}

void ofApp::draw()
{
    ofBackground(24);

    auto [srcW, srcH] = sourceSize();

    if (srcW <= 0 || srcH <= 0)
    {
        ofSetColor(255);
        // Distinguish "startup/no source" from "video backend accepted the file
        // but frames have not started arriving yet".
        std::string message = (mode == InputMode::Video) ? "Loading video " + sourceName +
                                                               " ... (if this persists, the file may be unplayable)"
                                                         : status;
        ofDrawBitmapString(message, 30, 40);
        gui.draw();
        return;
    }

    // Preserve aspect ratio for all source types while filling as much of the
    // window as possible.
    float scale = std::min(ofGetWidth() / srcW, ofGetHeight() / srcH);
    float offsetX = (ofGetWidth() - srcW * scale) / 2;
    float offsetY = (ofGetHeight() - srcH * scale) / 2;

    drawSource(offsetX, offsetY, srcW, srcH, scale);
    drawFaceOverlays(offsetX, offsetY, scale);
    drawHud();

    gui.draw();
}

void ofApp::drawSource(float offsetX, float offsetY, float srcW, float srcH, float scale)
{
    // Render whichever media source is currently active into the shared viewport.
    ofSetColor(255);
    if (mode == InputMode::Image)
    {
        image.draw(offsetX, offsetY, srcW * scale, srcH * scale);
    }
    else if (mode == InputMode::Video)
    {
        video.draw(offsetX, offsetY, srcW * scale, srcH * scale);
    }
    else
    {
        grabber.draw(offsetX, offsetY, srcW * scale, srcH * scale);
    }
}

void ofApp::drawFaceOverlays(float offsetX, float offsetY, float scale)
{
    for (size_t i = 0; i < faces.size(); i++)
    {
        const auto &face = faces[i];
        float x = offsetX + face.box.x * scale;
        float y = offsetY + face.box.y * scale;

        // The recognition pass always stores the nearest gallery identity.
        // The user-adjustable threshold is applied only for display so the UI
        // can be retuned without recomputing embeddings.
        bool recognized = i < matches.size() && matches[i].score >= matchThreshold;

        ofPushStyle();
        // Keep overlay-only style changes from leaking into later draw calls.
        ofNoFill();
        ofSetLineWidth(kOverlayLineWidth);
        if (recognized)
        {
            ofSetColor(kOverlayRecognizedR, kOverlayRecognizedG, kOverlayRecognizedB);
        }
        else
        {
            ofSetColor(kOverlayUnknownR, kOverlayUnknownG, kOverlayUnknownB);
        }
        ofDrawRectangle(x, y, face.box.width * scale, face.box.height * scale);
        ofFill();
        for (const auto &lm : face.landmarks)
        {
            ofDrawCircle(offsetX + lm.x * scale, offsetY + lm.y * scale, kLandmarkRadius);
        }
        ofPopStyle();

        std::string label = ofToString(face.confidence, 2);
        if (i < matches.size())
        {
            // A negative score means feature extraction failed for this face, so
            // there is no meaningful similarity number to render.
            std::string name = recognized ? matches[i].name : "unknown";
            label = matches[i].score < 0 ? name : name + " " + ofToString(matches[i].score, 2);
        }
        if (i < trackIds.size())
        {
            // Stable per-face ID from the tracker, prepended so identity swaps
            // between frames are easy to spot during video playback.
            label = "#" + ofToString(trackIds[i]) + " " + label;
        }
        if (i < liveStatus.size())
        {
            // Liveness flag (blink or mouth movement); nothing is appended
            // while the verdict is still pending so the label only claims what
            // has actually been observed.
            if (liveStatus[i].verdict == LivenessDetector::Verdict::Live)
            {
                label += " LIVE";
            }
            else if (liveStatus[i].verdict == LivenessDetector::Verdict::NoActivity)
            {
                label += " PHOTO?";
            }
        }
        // Draw labels just above each face box for quick identity scanning.
        ofDrawBitmapStringHighlight(label, x, y - kOverlayLabelOffsetY);
    }
}

void ofApp::drawHud()
{
    // Bottom-left HUD keeps frame stats readable regardless of source type.
    ofDrawBitmapStringHighlight("Faces: " + ofToString(faces.size()), kHudMarginLeft,
                                ofGetHeight() - kHudFacesBottomOffset);

    std::string info = sourceName + "  (" + ofToString(detectMillis, 1) + " ms/frame";
    if (mode != InputMode::Image)
    {
        info += ", " + ofToString(ofGetFrameRate(), 0) + " fps";
    }
    info += recognizer.hasGallery() ? ", gallery: " + ofToString(recognizer.personCount()) + " people" : ", no gallery";
    if (mode == InputMode::Video)
    {
        info += video.isPaused() ? ", paused — space resumes" : ", space pauses";
    }
    if (!webcamDevices.empty())
    {
        int index = ofClamp(static_cast<int>(webcamDeviceIndex.get()), 0, static_cast<int>(webcamDevices.size()) - 1);
        info += ", cam " + ofToString(index + 1) + "/" + ofToString(webcamDevices.size()) + ": " +
                webcamDevices[index].deviceName;
    }
    info += ")";
    ofDrawBitmapStringHighlight(info, kHudMarginLeft, ofGetHeight() - kHudInfoBottomOffset);
}

void ofApp::keyPressed(int key)
{
    if (key == 'o' || key == 'O')
    {
        auto result = ofSystemLoadDialog("Choose an image or video");
        if (result.bSuccess)
        {
            openPath(result.getPath());
        }
    }
    else if (key == ' ' && mode == InputMode::Video)
    {
        video.setPaused(!video.isPaused());
        ofLogNotice("facerec") << sourceName << (video.isPaused() ? ": paused" : ": resumed");
    }
    else if ((key == '[' || key == ']') && !webcamDevices.empty())
    {
        int current = ofClamp(static_cast<int>(webcamDeviceIndex.get()), 0, static_cast<int>(webcamDevices.size()) - 1);
        int delta = key == ']' ? 1 : -1;
        int count = static_cast<int>(webcamDevices.size());
        int next = (current + delta + count) % count;
        webcamDeviceIndex.set(next);
    }
}

void ofApp::dragEvent(ofDragInfo dragInfo)
{
    if (!dragInfo.files.empty())
    {
        ofLogNotice("facerec") << "dragged path " << dragInfo.files.front();
        openPath(dragInfo.files.front());
    }
}

void ofApp::onOpenImage()
{
    auto result = ofSystemLoadDialog("Choose an image");
    if (result.bSuccess)
    {
        loadImage(result.getPath());
    }
}

void ofApp::onOpenVideo()
{
    auto result = ofSystemLoadDialog("Choose a video file");
    if (result.bSuccess)
    {
        loadVideo(result.getPath());
    }
}

void ofApp::loadGallery(const std::string &path)
{
    int embeddings = recognizer.loadGallery(path, detector);
    ofLogNotice("facerec") << "gallery reload finished with " << embeddings << " embedding(s)";
    // A still image keeps its existing overlay until reprocessed, so refresh
    // immediately after the gallery changes.
    refreshStillImage();
}

void ofApp::onLoadGallery()
{
    auto result = ofSystemLoadDialog("Choose a gallery folder (one subfolder per person)", true /* folder selection */);
    if (result.bSuccess)
    {
        loadGallery(result.getPath());
    }
}

void ofApp::onWebcamToggle(bool &on)
{
    if (!on)
    {
        if (mode == InputMode::Webcam)
        {
            stopCurrentSource();
            status = "Webcam off. Open an image or video, or turn the webcam back on.";
            ofLogNotice("facerec") << "webcam stopped";
        }
        return;
    }
    refreshWebcamDevices(true);
    if (!startWebcam())
    {
        status = webcamDevices.empty() ? "No webcam devices found." : status;
        return;
    }
}

void ofApp::onWebcamDeviceIndexChanged(int &index)
{
    if (webcamDevices.empty() || index < 0 || index >= static_cast<int>(webcamDevices.size()))
    {
        return;
    }

    const ofVideoDevice &device = webcamDevices[index];
    updateWebcamDeviceControlLabel();
    ofLogNotice("facerec") << "selected webcam [" << device.id << "] " << device.deviceName;

    if (webcamOn)
    {
        startWebcam();
    }
    else
    {
        status = "Selected webcam: " + device.deviceName + ". Turn webcam on to use it.";
    }
}

void ofApp::onRefreshWebcams()
{
    bool webcamActive = mode == InputMode::Webcam && webcamOn;
    refreshWebcamDevices(true);

    if (webcamDevices.empty())
    {
        status = "No webcam devices found.";
        if (webcamActive)
        {
            stopCurrentSource();
        }
        return;
    }

    int index = ofClamp(static_cast<int>(webcamDeviceIndex.get()), 0, static_cast<int>(webcamDevices.size()) - 1);
    const ofVideoDevice &device = webcamDevices[index];
    status = "Selected webcam: " + device.deviceName;

    if (webcamActive)
    {
        startWebcam();
    }
}

void ofApp::onScoreThreshold(float &value)
{
    detector.setScoreThreshold(value);
    ofLogNotice("facerec") << "detector threshold changed to " << ofToString(value, 2);
    // Live sources pick the new threshold up on the next frame. Still images
    // have to be reprocessed explicitly because their last frame is cached.
    refreshStillImage();
}

void ofApp::onTrackingToggle(bool &on)
{
    // Restart from a clean slate in both directions so stale IDs never
    // linger on screen (e.g. while a video is paused) and re-enabling starts
    // a fresh numbering.
    tracker.reset();
    trackIds.clear();
    // The tracker restarts ID numbering from 1, so blink histories keyed by
    // the old IDs would be inherited by unrelated faces — drop them too.
    liveness.reset();
    liveStatus.clear();
    ofLogNotice("facerec") << (on ? "tracking enabled" : "tracking disabled");
}

void ofApp::onLivenessToggle(bool &on)
{
    // Blink/mouth evidence collected earlier would be stale by the time
    // liveness is re-enabled, so both directions restart the observation.
    liveness.reset();
    liveStatus.clear();
    ofLogNotice("facerec") << (on ? "liveness enabled" : "liveness disabled");
}
