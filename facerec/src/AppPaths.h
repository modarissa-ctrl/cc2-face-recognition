/**
 * @file AppPaths.h
 * @brief Shared relative paths for app data files under `bin/data`.
 */

#pragma once

namespace AppPaths
{

/**
 * @brief Relative path to the YuNet detector model inside `bin/data`.
 */
static constexpr const char kYunetModel[] = "models/face_detection_yunet_2023mar.onnx";

/**
 * @brief Relative path to the SFace recognizer model inside `bin/data`.
 */
static constexpr const char kSfaceModel[] = "models/face_recognition_sface_2021dec.onnx";

/**
 * @brief Relative path to the default recognition gallery folder.
 */
static constexpr const char kGalleryDir[] = "gallery";

} // namespace AppPaths
