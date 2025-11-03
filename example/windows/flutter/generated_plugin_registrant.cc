//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <file_selector_windows/file_selector_windows.h>
#include <pose_detection_tflite/pose_detection_tflite_plugin.h>

void RegisterPlugins(flutter::PluginRegistry* registry) {
  FileSelectorWindowsRegisterWithRegistrar(
      registry->GetRegistrarForPlugin("FileSelectorWindows"));
  PoseDetectionTflitePluginRegisterWithRegistrar(
      registry->GetRegistrarForPlugin("PoseDetectionTflitePlugin"));
}
