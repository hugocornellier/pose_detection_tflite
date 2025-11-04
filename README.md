# pose_detection_tflite

A pure Dart/Flutter implementation of Google's MediaPipe pose detection and facial landmark models using TensorFlow Lite. 
This package provides on-device face and landmark detection with minimal dependencies, just TensorFlow Lite and image.

## Quick Start

```dart
import 'dart:io';
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

Future main() async {
  // 1. initialize
  final detector = PoseDetector();
  await detector.initialize(
    options: const PoseOptions(
      mode: PoseMode.boxesAndLandmarks,
      landmarkModel: PoseLandmarkModel.heavy,
    ),
  );

  // 2. detect
  final imageBytes = await File('path/to/image.jpg').readAsBytes();
  final results = await detector.detect(imageBytes);

  // 3. access results
  for (final pose in results) {
    final bbox = pose.bboxPx;
    print('Bounding box: (${bbox.left}, ${bbox.top}) → (${bbox.right}, ${bbox.bottom})');

    if (pose.hasLandmarks) {
      for (final lm in pose.landmarks) {
        print('${lm.type}: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)}) vis=${lm.visibility.toStringAsFixed(2)}');
      }
    }
  }

  // 4. clean-up
  await detector.dispose();
}
```

## Pose Detection Modes

This package supports two operation modes that determine what data is returned:

| Mode                            | Description                                 | Output                        |
| ------------------------------- | ------------------------------------------- | ----------------------------- |
| **boxesAndLandmarks** (default) | Full two-stage detection (YOLO + BlazePose) | Bounding boxes + 33 landmarks |
| **boxes**                       | Fast YOLO-only detection                    | Bounding boxes only           |