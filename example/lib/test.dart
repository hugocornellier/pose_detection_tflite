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
  final Uint8List imageBytes = await File('path/to/image.jpg').readAsBytes();
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