/**
 * @file ofApp.h
 * @brief Interactive openFrameworks application for face detection and recognition.
 */

#pragma once

#include <utility>

#include "ofMain.h"
#include "ofxGui.h"

#include "detection/FaceDetector.h"
#include "recognition/FaceRecognizer.h"
#include "tracking/FaceTracker.h"
#include "liveness/LivenessDetector.h"

/**
 * @brief Interactive face recognition application.
 *
 * The app supports still images, video playback, and a live webcam. Frames are
 * detected with YuNet, optionally recognized with SFace against a folder-based
 * gallery, tracked across video/webcam frames with stable per-face IDs,
 * checked for liveness via blinks and mouth movement (LIVE / PHOTO? flags),
 * and then rendered with debugging overlays and timing information.
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

    FaceDetector detector;            //< YuNet model instance used for face detection.
    FaceRecognizer recognizer;        //< SFace model instance used for face recognition.
    FaceTracker tracker;              //< Frame-to-frame tracker assigning stable face IDs.
    LivenessDetector liveness;        //< Blink/mouth-movement live-photo classifier for tracked faces.
    InputMode mode = InputMode::None; //< Current source type feeding frames into the pipeline.

    ofImage image;                            //< Loaded still image.
    ofVideoPlayer video;                      //< Loaded video source.
    ofVideoGrabber grabber;                   //< Live webcam feed.
    std::vector<ofVideoDevice> webcamDevices; //< Enumerated webcam devices available for selection.
    std::string sourceName;                   //< Name of the current media source.

    std::vector<FaceDetection> faces; //< Detected faces in the current frame.
    std::vector<FaceMatch> matches;   //< Recognition results for the detected faces in the current frame.
    std::vector<int> trackIds;        //< Stable track IDs for the detected faces, empty when not tracking.
    std::vector<LivenessDetector::Status> liveStatus; //< Liveness verdicts per face, empty when liveness is off.
    float detectMillis = 0.0f;                        //< Time taken for the last detection in milliseconds.
    uint64_t lastLogMillis = 0;                       //< Timestamp of the last log message.
    std::string status;                               //< Current status message.

    ofxPanel gui;                                                 //< GUI panel containing all controls.
    ofxButton openImageButton;                                    //< Button to open an image file.
    ofxButton openVideoButton;                                    //< Button to open a video file.
    ofxButton loadGalleryButton;                                  //< Button to load a face recognition gallery.
    ofxButton refreshWebcamsButton;                               //< Button to refresh the webcam device list.
    ofParameter<bool> webcamOn{"webcam", false};                  //< Toggle for webcam input.
    ofParameter<int> webcamDeviceIndex{"webcam device", 0, 0, 0}; //< Index into `webcamDevices`.
    ofParameter<bool> trackingOn{"tracking", true};               //< Toggle for stable face IDs on video/webcam.
    ofParameter<bool> livenessOn{"liveness", true}; //< Toggle for blink/mouth LIVE-PHOTO? flags (needs tracking).
    ofParameter<float> scoreThreshold{"conf threshold", 0.6f, 0.05f,
                                      0.95f}; //< Confidence threshold for face detection.
    ofParameter<float> matchThreshold{"match threshold", FaceRecognizer::kDefaultMatchThreshold, 0.0f,
                                      1.0f}; //< Threshold for face recognition matches.

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
    void stopCurrentSource(bool keepWebcamToggle = false);

    /**
     * @brief Enumerate webcam devices and update the GUI selector range.
     * @param keepSelection Keep the current selection when the same device is still present.
     */
    void refreshWebcamDevices(bool keepSelection);

    /**
     * @brief Start webcam capture using the currently selected device.
     * @return `true` when the selected device opened successfully.
     */
    bool startWebcam();

    /**
     * @brief Update the webcam selector label to include the selected device name.
     */
    void updateWebcamDeviceControlLabel();

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
     * @brief Draw the active image, video, or webcam source at the provided bounds.
     * @param offsetX Horizontal destination offset in window coordinates.
     * @param offsetY Vertical destination offset in window coordinates.
     * @param srcW Source width in pixels.
     * @param srcH Source height in pixels.
     * @param scale Source-to-window scale factor.
     */
    void drawSource(float offsetX, float offsetY, float srcW, float srcH, float scale);

    /**
     * @brief Draw detection boxes, landmarks, and identity labels.
     * @param offsetX Horizontal destination offset in window coordinates.
     * @param offsetY Vertical destination offset in window coordinates.
     * @param scale Source-to-window scale factor.
     */
    void drawFaceOverlays(float offsetX, float offsetY, float scale);

    /**
     * @brief Draw source, performance, and gallery status text.
     */
    void drawHud();

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
     * @brief GUI callback for changing the selected webcam device.
     * @param index New index into `webcamDevices`.
     */
    void onWebcamDeviceIndexChanged(int &index);

    /**
     * @brief GUI callback for refreshing the available webcam list.
     */
    void onRefreshWebcams();

    /**
     * @brief GUI callback for updating the YuNet confidence threshold.
     * @param value New threshold value.
     */
    void onScoreThreshold(float &value);

    /**
     * @brief GUI callback for enabling or disabling face tracking.
     * @param on New tracking toggle state.
     */
    void onTrackingToggle(bool &on);

    /**
     * @brief GUI callback for enabling or disabling liveness detection.
     * @param on New liveness toggle state.
     */
    void onLivenessToggle(bool &on);
};
