#ifndef POSE_DETECTION_TFLITE_PUBLIC_PLUGIN_H_
#define POSE_DETECTION_TFLITE_PUBLIC_PLUGIN_H_

#include <flutter_windows.h>

#ifdef FLUTTER_PLUGIN_IMPL
#define FLUTTER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FLUTTER_PLUGIN_EXPORT __declspec(dllimport)
#endif

FLUTTER_PLUGIN_EXPORT void PoseDetectionTflitePluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar);

#endif  // POSE_DETECTION_TFLITE_PUBLIC_PLUGIN_H_
