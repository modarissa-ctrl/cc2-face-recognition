/**
 * @file ofApp.h
 * @brief Interactive openFrameworks application for face detection and recognition.
 */

#pragma once

#include <utility>

#include "ofMain.h"
#include "ofxGui.h"

#include "FaceDetector.h"
#include "FaceRecognizer.h"

/**
 * @brief Interactive face recognition application.
 *
 * The app supports still images, video playback, and a live webcam. Frames are
 * detected with YuNet, optionally recognized with SFace against a folder-based
 * gallery, and then rendered with debugging overlays and timing information.
 *
 * Headless utilities such as `--selftest`, `--detect`, and `--identify` are
 * handled in `main()` before any OpenGL window is created.
 */
class ofApp : public ofBaseApp
{
  public:
    /**
     * @brief Initialize models, GUI controls, and optional startup media.
     */
    void setup() override;
    /**
     * @brief Advance the active source and run per-frame detection.
     */
    void update() override;
    /**
     * @brief Draw the active source, overlays, performance text, and GUI.
     */
    void draw() override;
    /**
     * @brief Handle keyboard shortcuts such as opening files or pausing video.
     * @param key Pressed key code.
     */
    void keyPressed(int key) override;
    /**
     * @brief Handle drag-and-drop media input.
     * @param dragInfo Dropped file metadata from openFrameworks.
     */
    void dragEvent(ofDragInfo dragInfo) override;

    /**
     * @brief Command-line arguments forwarded from `main()`.
     */
    std::vector<std::string> args;

  private:
    /**
     * @brief Runtime source type that currently feeds frames into the pipeline.
     */
    enum class InputMode
    {
        None,
        Image,
        Video,
        Webcam
    };

    FaceDetector detector;
    FaceRecognizer recognizer;
    InputMode mode = InputMode::None;

    ofImage image;
    ofVideoPlayer video;
    ofVideoGrabber grabber;
    std::string sourceName;

    std::vector<FaceDetection> faces;
    std::vector<FaceMatch> matches;
    float detectMillis = 0.0f;
    uint64_t lastLogMillis = 0;
    std::string status;

    ofxPanel gui;
    ofxButton openImageButton;
    ofxButton openVideoButton;
    ofxButton loadGalleryButton;
    ofParameter<bool> webcamOn{"webcam", false};
    ofParameter<float> scoreThreshold{"conf threshold", 0.6f, 0.05f, 0.95f};
    ofParameter<float> matchThreshold{"match threshold", FaceRecognizer::kDefaultMatchThreshold, 0.0f, 1.0f};

    /**
     * @brief Open a path and dispatch it to image or video loading.
     * @param path Absolute or data-relative media path.
     * @return `true` when the source was opened successfully.
     */
    bool openPath(const std::string &path);
    /**
     * @brief Load a still image and immediately process it.
     * @param path Image file path.
     * @return `true` when the image was loaded successfully.
     */
    bool loadImage(const std::string &path);
    /**
     * @brief Load a video source and start playback.
     * @param path Video file path.
     * @return `true` when the video backend accepted the file.
     */
    bool loadVideo(const std::string &path);
    /**
     * @brief Tear down the currently active media source and reset state.
     */
    void stopCurrentSource();
    /**
     * @brief Run detection and optional recognition on one frame.
     * @param pixels Frame pixels from the active source.
     */
    void detectFrame(const ofPixels &pixels);
    /**
     * @brief Load or reload the recognition gallery.
     * @param path Folder containing one subfolder per person.
     */
    void loadGallery(const std::string &path);

    /**
     * @brief Report whether the current source is a loaded still image.
     * @return `true` when an image is active and allocated.
     */
    bool hasStillImage() const
    {
        return mode == InputMode::Image && image.isAllocated();
    }
    /**
     * @brief Re-run detection for the current still image if one is active.
     */
    void refreshStillImage();

    /**
     * @brief Current source dimensions in pixels.
     * @return Width and height, or `{0, 0}` when no source is producing frames.
     */
    std::pair<float, float> sourceSize() const;

    /**
     * @brief GUI callback for opening an image file.
     */
    void onOpenImage();
    /**
     * @brief GUI callback for opening a video file.
     */
    void onOpenVideo();
    /**
     * @brief GUI callback for choosing a gallery directory.
     */
    void onLoadGallery();
    /**
     * @brief GUI callback for enabling or disabling the webcam.
     * @param on New webcam toggle state.
     */
    void onWebcamToggle(bool &on);
    /**
     * @brief GUI callback for updating the YuNet confidence threshold.
     * @param value New threshold value.
     */
    void onScoreThreshold(float &value);
};
