#include "include/face_detection_tflite/face_detection_tflite_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "face_detection_tflite_plugin.h"

void PoseDetectionTflitePluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  face_detection_tflite::PoseDetectionTflitePlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
