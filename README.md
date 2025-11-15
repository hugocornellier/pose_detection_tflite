# pose_detection_tflite

A pure Dart/Flutter implementation of Google's MediaPipe pose detection and facial landmark models using TensorFlow Lite. 
This package provides on-device, multi-person pose detection with minimal dependencies, just TensorFlow Lite and image.

![Example Screenshot](assets/screenshots/ex1.png)

## Quick Start

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

Future main() async {
  // 1. initialize
  final PoseDetector detector = PoseDetector(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: PoseLandmarkModel.heavy,
  );
  await detector.initialize();

  // 2. detect
  final Uint8List imageBytes = await File('image.jpg').readAsBytes();
  final List<Pose> results = await detector.detect(imageBytes);

  // 3. access results
  for (final Pose pose in results) {
    final BoundingBox bbox = pose.boundingBox;
    print('Bounding box: (${bbox.left}, ${bbox.top}) → (${bbox.right}, ${bbox.bottom})');

    if (pose.hasLandmarks) {
      // iterate through landmarks
      for (final PoseLandmark lm in pose.landmarks) {
        print('${lm.type}: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)}) vis=${lm.visibility.toStringAsFixed(2)}');
      }

      // access individual landmarks
      // see "Pose Landmark Types" section in README for full list of landmarks
      final PoseLandmark? leftKnee = pose.getLandmark(PoseLandmarkType.leftKnee);
      if (leftKnee != null) {
        print('Left knee visibility: ${leftKnee.visibility.toStringAsFixed(2)}');
      }
    }
  }

  // 4. clean-up
  await detector.dispose();
}
```

Refer to the [sample code](https://pub.dev/packages/pose_detection_tflite/example) on the pub.dev example tab for a more in-depth example.

## Pose Detection Modes

This package supports two operation modes that determine what data is returned:

| Mode                            | Description                                 | Output                        |
| ------------------------------- | ------------------------------------------- | ----------------------------- |
| **boxesAndLandmarks** (default) | Full two-stage detection (YOLO + BlazePose) | Bounding boxes + 33 landmarks |
| **boxes**                       | Fast YOLO-only detection                    | Bounding boxes only           |

### Use boxes-only mode for faster detection

When you only need to detect where people are (without body landmarks), use `PoseMode.boxes` for better performance:

```dart
final PoseDetector detector = PoseDetector(
  mode: PoseMode.boxes,  // Skip landmark detection
);
await detector.initialize();

final List<Pose> results = await detector.detect(imageBytes);
for (final Pose pose in results) {
  print('Person detected at: ${pose.boundingBox}');
  print('Detection confidence: ${pose.score.toStringAsFixed(2)}');
  // pose.hasLandmarks will be false
}
```

## Pose Landmark Models

Choose the model that fits your performance needs:

| Model | Speed | Accuracy |
|-------|-------|----------|
| **lite** | Fastest | Good |
| **full** | Balanced | Better |
| **heavy** | Slowest | Best |

## Pose Landmark Types

Every pose contains up to 33 landmarks that align with the BlazePose specification:

- nose
- leftEyeInner
- leftEye
- leftEyeOuter
- rightEyeInner
- rightEye
- rightEyeOuter
- leftEar
- rightEar
- mouthLeft
- mouthRight
- leftShoulder
- rightShoulder
- leftElbow
- rightElbow
- leftWrist
- rightWrist
- leftPinky
- rightPinky
- leftIndex
- rightIndex
- leftThumb
- rightThumb
- leftHip
- rightHip
- leftKnee
- rightKnee
- leftAnkle
- rightAnkle
- leftHeel
- rightHeel
- leftFootIndex
- rightFootIndex

```dart
// Example - how to access specific landmarks
// PoseLandmarkType can be any of the 33 landmarks listed above.
final PoseLandmark? leftHip = pose.getLandmark(PoseLandmarkType.leftHip);
if (leftHip != null && leftHip.visibility > 0.5) {
    // Pixel coordinates in original image space
    print('Left hip position: (${leftHip.x}, ${leftHip.y})');
    
    // Depth information (relative z-coordinate)
    print('Left hip depth: ${leftHip.z}');
}
```

## Advanced Usage

### Processing pre-decoded images

If you already have an `Image` object from the `image` package, use `detectOnImage()` to skip re-decoding:

```dart
import 'package:image/image.dart' as img;

final img.Image image = img.decodeImage(imageBytes)!;
final List<Pose> results = await detector.detectOnImage(image);
```

### Multi-person detection

The detector automatically handles multiple people in a single image:

```dart
final List<Pose> results = await detector.detect(imageBytes);
print('Detected ${results.length} people');

for (int i = 0; i < results.length; i++) {
  final Pose pose = results[i];
  print('Person ${i + 1}:');
  print('Bounding box: ${pose.boundingBox}');
  print('Confidence: ${pose.score.toStringAsFixed(2)}');
  print('Landmarks: ${pose.landmarks.length}');
}
```

### Camera/video stream processing

For real-time camera processing, reuse the same detector instance:

```dart
final detector = PoseDetector(
  landmarkModel: PoseLandmarkModel.lite,  // Use lite for better FPS
  detectorConf: 0.6,
);
await detector.initialize();

// Process each frame
void processFrame(Uint8List frameBytes) async {
  final results = await detector.detect(frameBytes);
  // Update UI with results
}

await detector.dispose();
```
