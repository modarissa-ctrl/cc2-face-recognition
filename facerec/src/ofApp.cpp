/**
 * @file ofApp.cpp
 * @brief Interactive application flow for image, video, and webcam recognition.
 */

#include "ofApp.h"
#include "AppPaths.h"

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
    ofSetWindowTitle("facerec — M3 recognition");

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
    gui.add(scoreThreshold);
    gui.add(matchThreshold);
    gui.add(loadGalleryButton.setup("load gallery..."));
    openImageButton.addListener(this, &ofApp::onOpenImage);
    openVideoButton.addListener(this, &ofApp::onOpenVideo);
    loadGalleryButton.addListener(this, &ofApp::onLoadGallery);
    webcamOn.addListener(this, &ofApp::onWebcamToggle);
    scoreThreshold.addListener(this, &ofApp::onScoreThreshold);

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

void ofApp::stopCurrentSource()
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
        webcamOn.setWithoutEventNotifications(false);
    }
    mode = InputMode::None;
    faces.clear();
    matches.clear();
    detectMillis = 0.0f;
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
            ofLogNotice("facerec") << sourceName << ": " << faces.size() << " face(s), " << ofToString(detectMillis, 1)
                                   << " ms/frame, " << ofToString(ofGetFrameRate(), 0) << " fps";
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
    stopCurrentSource();
    grabber.setDesiredFrameRate(30);
    grabber.setup(1280, 720);
    if (!grabber.isInitialized())
    {
        status = "Could not open the webcam.";
        ofLogError("facerec") << status;
        webcamOn.setWithoutEventNotifications(false);
        return;
    }
    mode = InputMode::Webcam;
    sourceName = "webcam";
    status.clear();
    lastLogMillis = 0;
    ofLogNotice("facerec") << "webcam started at 1280x720";
}

void ofApp::onScoreThreshold(float &value)
{
    detector.setScoreThreshold(value);
    ofLogNotice("facerec") << "detector threshold changed to " << ofToString(value, 2);
    // Live sources pick the new threshold up on the next frame. Still images
    // have to be reprocessed explicitly because their last frame is cached.
    refreshStillImage();
}
