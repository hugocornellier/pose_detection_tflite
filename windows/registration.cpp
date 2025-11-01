#include "include/face_detection_tflite/face_detection_tflite_plugin.h"
#include "face_detection_tflite_plugin.h"
#include <flutter/plugin_registrar_windows.h>

void PoseDetectionTflitePluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  face_detection_tflite::PoseDetectionTflitePlugin::RegisterWithRegistrar(cpp_registrar);
}
