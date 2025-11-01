#ifndef FLUTTER_PLUGIN_FACE_DETECTION_TFLITE_PLUGIN_H_
#define FLUTTER_PLUGIN_FACE_DETECTION_TFLITE_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace face_detection_tflite {

class PoseDetectionTflitePlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  PoseDetectionTflitePlugin();

  virtual ~PoseDetectionTflitePlugin();

  // Disallow copy and assign.
  PoseDetectionTflitePlugin(const PoseDetectionTflitePlugin&) = delete;
  PoseDetectionTflitePlugin& operator=(const PoseDetectionTflitePlugin&) = delete;

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace face_detection_tflite

#endif  // FLUTTER_PLUGIN_FACE_DETECTION_TFLITE_PLUGIN_H_
