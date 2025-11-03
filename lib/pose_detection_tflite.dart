import 'dart:io';
import 'dart:math' as math;
import 'dart:ffi' as ffi;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'package:path/path.dart' as p;

// ============================================================================
// POSE DETECTION LIBRARY - Single File Implementation
// ============================================================================

/// Main pose detection class that handles the two-stage pipeline:
/// 1. Pose Detection - Detects person bounding boxes
/// 2. Pose Landmark - Estimates keypoints within detected regions
class PoseDetector {
  Interpreter? _detectorInterpreter;
  Interpreter? _landmarkInterpreter;

  bool _isInitialized = false;
  PoseModelComplexity _complexity = PoseModelComplexity.lite;

  // Static TFLite library reference
  static ffi.DynamicLibrary? _tfliteLib;
  static List<List<double>>? _anchors;

  static void _generateAnchors() {
    if (_anchors != null) return;
    final anchors = <List<double>>[];
    const int numSteps1 = 28;
    final double step1 = 1.0 / numSteps1;
    final double offset1 = step1 / 2;
    for (int y = 0; y < numSteps1; y++) {
      for (int x = 0; x < numSteps1; x++) {
        final double anchorX = offset1 + x * step1;
        final double anchorY = offset1 + y * step1;
        anchors.add([anchorX, anchorY, 1.0, 1.0]);
        anchors.add([anchorX, anchorY, 1.0, 1.0]);
      }
    }
    const int numSteps2 = 14;
    final double step2 = 1.0 / numSteps2;
    final double offset2 = step2 / 2;
    for (int y = 0; y < numSteps2; y++) {
      for (int x = 0; x < numSteps2; x++) {
        final double anchorX = offset2 + x * step2;
        final double anchorY = offset2 + y * step2;
        anchors.add([anchorX, anchorY, 1.0, 1.0]);
        anchors.add([anchorX, anchorY, 1.0, 1.0]);
      }
    }
    const int numSteps3 = 7;
    final double step3 = 1.0 / numSteps3;
    final double offset3 = step3 / 2;
    for (int y = 0; y < numSteps3; y++) {
      for (int x = 0; x < numSteps3; x++) {
        final double anchorX = offset3 + x * step3;
        final double anchorY = offset3 + y * step3;
        for (int k = 0; k < 6; k++) {
          anchors.add([anchorX, anchorY, 1.0, 1.0]);
        }
      }
    }
    _anchors = anchors;
    print('anchors=${_anchors!.length}');
  }

  /// Ensure TensorFlow Lite library is loaded
  static Future<void> _ensureTFLiteLoaded() async {
    if (_tfliteLib != null) return;

    final exe = File(Platform.resolvedExecutable);
    final exeDir = exe.parent;

    late final List<String> candidates;
    late final String hint;

    if (Platform.isWindows) {
      candidates = [
        p.join(exeDir.path, 'libtensorflowlite_c-win.dll'),
        'libtensorflowlite_c-win.dll',
      ];
      hint =
      'Make sure your Windows plugin CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries ".../libtensorflowlite_c-win.dll" PARENT_SCOPE)\n'
          'so Flutter copies it next to the app EXE.';
    } else if (Platform.isLinux) {
      candidates = [
        p.join(exeDir.path, 'lib', 'libtensorflowlite_c-linux.so'),
        'libtensorflowlite_c-linux.so',
      ];
      hint =
      'Ensure linux/CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries "../assets/bin/libtensorflowlite_c-linux.so" PARENT_SCOPE)\n'
          'so Flutter copies it into bundle/lib/.';
    } else if (Platform.isMacOS) {
      final contents = exeDir.parent;
      candidates = [
        p.join(contents.path, 'Resources', 'libtensorflowlite_c-mac.dylib'),
        'libtensorflowlite_c-mac.dylib',
      ];
      hint = 'Expected in app bundle Resources, or resolvable by name.';
    } else {
      _tfliteLib = ffi.DynamicLibrary.process();
      return;
    }

    final tried = <String>[];
    for (final c in candidates) {
      try {
        if (c.contains(p.separator)) {
          if (!File(c).existsSync()) {
            tried.add(c);
            continue;
          }
        }
        _tfliteLib = ffi.DynamicLibrary.open(c);
        return;
      } catch (_) {
        tried.add(c);
      }
    }

    throw ArgumentError(
      'Failed to locate TensorFlow Lite C library.\n'
          'Tried:\n - ${tried.join('\n - ')}\n\n$hint',
    );
  }

  /// Initialize the pose detector with specified complexity
  Future<void> initialize({
    PoseModelComplexity complexity = PoseModelComplexity.lite,
  }) async {
    if (_isInitialized) {
      await dispose();
    }

    // Load TFLite library first
    await _ensureTFLiteLoaded();

    _complexity = complexity;

    // Load detector model directly from assets
    _detectorInterpreter = await _loadModelFromAssets(
      'assets/models/pose_detection.tflite',
    );

    // Load landmark model based on complexity
    final landmarkPath = _getLandmarkModelPath(complexity);
    _landmarkInterpreter = await _loadModelFromAssets(landmarkPath);
    _generateAnchors();

    _isInitialized = true;

  }

  String _getLandmarkModelPath(PoseModelComplexity complexity) {
    switch (complexity) {
      case PoseModelComplexity.lite:
        return 'assets/models/pose_landmark_lite.tflite';
      case PoseModelComplexity.full:
        return 'assets/models/pose_landmark_full.tflite';
      case PoseModelComplexity.heavy:
        return 'assets/models/pose_landmark_heavy.tflite';
    }
  }

  Future<Interpreter> _loadModelFromAssets(String assetPath) async {
    return await Interpreter.fromAsset(assetPath);
  }

  /// Detect pose from an image file
  Future<PoseDetectionResult?> detectPose(File imageFile) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }

    // Load and decode image
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes);
    if (image == null) return null;

    return detectPoseFromImage(image);
  }

  /// Detect pose from an Image object
  Future<PoseDetectionResult?> detectPoseFromImage(img.Image image) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }

    final fullFrame = PoseDetection(
      xCenter: 0.5,
      yCenter: 0.5,
      width: 1.0,
      height: 1.0,
      score: 1.0,
      keypoints: const [],
    );

    final roiImage = img.copyResize(image, width: 256, height: 256);
    print('roi_size=${roiImage.width}x${roiImage.height}');

    final landmarks = await _runLandmarkDetection(roiImage);
    print('lm_score=${landmarks.score.toStringAsFixed(3)}');
    if (landmarks.score < 0.5) {
      print('lm_reject_low_score');
      return null;
    }

    final transformedLandmarks = _transformLandmarks(
      landmarks,
      fullFrame,
      image.width,
      image.height,
    );

    if (transformedLandmarks.isNotEmpty) {
      final l0 = transformedLandmarks.first;
      print('lm0=${l0.x.toStringAsFixed(3)},${l0.y.toStringAsFixed(3)} vis=${l0.visibility.toStringAsFixed(3)}');
    }

    return PoseDetectionResult(
      landmarks: transformedLandmarks,
      detection: fullFrame,
      imageWidth: image.width,
      imageHeight: image.height,
    );
  }


  /// Stage 1: Run pose detection model
  Future<List<PoseDetection>> _runPoseDetection(img.Image image) async {
    // Prepare input: resize to 224x224
    final inputImage = img.copyResize(image, width: 224, height: 224);
    final input = _imageToFloat32List(inputImage, 224, 224);

    // Prepare outputs
    // Output 0: [1, 2254, 12] - bounding boxes and keypoints
    // Output 1: [1, 2254, 1] - confidence scores
    final outputBoxes = _reshapeToTensor3D(
      List.filled(1 * 2254 * 12, 0.0),
      1,
      2254,
      12,
    );
    final outputScores = _reshapeToTensor3D(
      List.filled(1 * 2254 * 1, 0.0),
      1,
      2254,
      1,
    );

    // Run inference
    _detectorInterpreter!.runForMultipleInputs(
      [input],
      {
        0: outputBoxes,
        1: outputScores,
      },
    );

    // Parse detections
    return _parseDetections(outputBoxes, outputScores);
  }

  List<PoseDetection> _parseDetections(
      List<dynamic> boxes,
      List<dynamic> scores,
      ) {
    double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
    double _clamp01(double v) => v.isNaN ? 0.0 : v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);

    final detections = <PoseDetection>[];
    const double scoreThreshold = 0.3;

    for (int i = 0; i < 2254; i++) {
      final rawScore = scores[0][i][0] as double;
      final score = _sigmoid(rawScore);
      if (score <= scoreThreshold) continue;

      final yCenter = _clamp01(boxes[0][i][0] as double);
      final xCenter = _clamp01(boxes[0][i][1] as double);
      final height = _clamp01(boxes[0][i][2] as double);
      final width  = _clamp01(boxes[0][i][3] as double);

      final keypoints = <PoseKeypoint>[];
      for (int k = 0; k < 4; k++) {
        final kx = _clamp01(boxes[0][i][4 + k * 2] as double);
        final ky = _clamp01(boxes[0][i][5 + k * 2] as double);
        keypoints.add(PoseKeypoint(x: kx, y: ky, score: 1.0));
      }

      detections.add(PoseDetection(
        xCenter: xCenter,
        yCenter: yCenter,
        width: width,
        height: height,
        score: score,
        keypoints: keypoints,
      ));
    }

    final kept = _nonMaxSuppression(detections);
    print('detections_kept=${kept.length}');
    return kept;
  }


  List<PoseDetection> _nonMaxSuppression(
      List<PoseDetection> detections, {
        double iouThreshold = 0.3,
      }) {
    if (detections.isEmpty) return [];

    // Sort by score descending
    detections.sort((a, b) => b.score.compareTo(a.score));

    final selected = <PoseDetection>[];
    final suppressed = <bool>[];

    for (int i = 0; i < detections.length; i++) {
      suppressed.add(false);
    }

    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;

      selected.add(detections[i]);

      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;

        final iou = _calculateIOU(detections[i], detections[j]);
        if (iou > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return selected;
  }

  double _calculateIOU(PoseDetection a, PoseDetection b) {
    final aLeft = a.xCenter - a.width / 2;
    final aRight = a.xCenter + a.width / 2;
    final aTop = a.yCenter - a.height / 2;
    final aBottom = a.yCenter + a.height / 2;

    final bLeft = b.xCenter - b.width / 2;
    final bRight = b.xCenter + b.width / 2;
    final bTop = b.yCenter - b.height / 2;
    final bBottom = b.yCenter + b.height / 2;

    final intersectLeft = math.max(aLeft, bLeft);
    final intersectRight = math.min(aRight, bRight);
    final intersectTop = math.max(aTop, bTop);
    final intersectBottom = math.min(aBottom, bBottom);

    if (intersectRight < intersectLeft || intersectBottom < intersectTop) {
      return 0.0;
    }

    final intersectArea =
        (intersectRight - intersectLeft) * (intersectBottom - intersectTop);
    final aArea = a.width * a.height;
    final bArea = b.width * b.height;
    final unionArea = aArea + bArea - intersectArea;

    return intersectArea / unionArea;
  }

  /// Extract ROI from image with padding
  img.Image _extractAndPadROI(img.Image image, PoseDetection detection) {
    const paddingFactor = 1.8;
    final w = (detection.width * paddingFactor).clamp(0.1, 1.0);
    final h = (detection.height * paddingFactor).clamp(0.1, 1.0);
    final cx = detection.xCenter.clamp(0.0, 1.0);
    final cy = detection.yCenter.clamp(0.0, 1.0);

    final left = ((cx - w / 2) * image.width).round();
    final top = ((cy - h / 2) * image.height).round();
    final rw = (w * image.width).round();
    final rh = (h * image.height).round();

    final cl = left.clamp(0, image.width);
    final ct = top.clamp(0, image.height);
    final cr = (left + rw).clamp(0, image.width);
    final cb = (top + rh).clamp(0, image.height);

    final cropW = (cr - cl).clamp(1, image.width);
    final cropH = (cb - ct).clamp(1, image.height);

    final cropped = img.copyCrop(
      image,
      x: cl,
      y: ct,
      width: cropW,
      height: cropH,
    );

    return img.copyResize(cropped, width: 256, height: 256);
  }


  /// Stage 2: Run landmark detection model
  Future<PoseLandmarks> _runLandmarkDetection(img.Image roiImage) async {
    // Prepare input: already 256x256
    final input = _imageToFloat32List(roiImage, 256, 256);

    // Prepare outputs based on model inspection:
    // [0] Shape: [1, 195] - landmarks (33 points × 3 coords + 33 × 3 world coords)
    // [1] Shape: [1, 1] - pose score
    // [2] Shape: [1, 256, 256, 1] - segmentation mask
    // [3] Shape: [1, 64, 64, 39] - heatmap
    // [4] Shape: [1, 117] - world landmarks (33 × 3 + visibility)
    final outputLandmarks = [List.filled(195, 0.0)];
    final outputScore = [[0.0]];
    final outputMask = _reshapeToTensor4D(
      List.filled(1 * 256 * 256 * 1, 0.0),
      1,
      256,
      256,
      1,
    );
    final outputHeatmap = _reshapeToTensor4D(
      List.filled(1 * 64 * 64 * 39, 0.0),
      1,
      64,
      64,
      39,
    );
    final outputWorld = [List.filled(117, 0.0)];

    // Run inference
    _landmarkInterpreter!.runForMultipleInputs(
      [input],
      {
        0: outputLandmarks,
        1: outputScore,
        2: outputMask,
        3: outputHeatmap,
        4: outputWorld,
      },
    );

    // Parse landmarks
    return _parseLandmarks(outputLandmarks, outputScore, outputWorld);
  }

  PoseLandmarks _parseLandmarks(
      List<dynamic> landmarksData,
      List<dynamic> scoreData,
      List<dynamic> worldData,
      ) {
    double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
    double _clamp01(double v) => v.isNaN ? 0.0 : v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);

    final score = _sigmoid(scoreData[0][0] as double);
    final raw = landmarksData[0] as List<dynamic>;
    final lm = <PoseLandmark>[];

    for (int i = 0; i < 33; i++) {
      final base = i * 5;
      final x = _clamp01((raw[base + 0] as double) / 256.0);
      final y = _clamp01((raw[base + 1] as double) / 256.0);
      final z = raw[base + 2] as double;
      final visibility = _sigmoid(raw[base + 3] as double);
      final presence = _sigmoid(raw[base + 4] as double);
      final vis = (visibility * presence).clamp(0.0, 1.0);

      lm.add(PoseLandmark(
        type: PoseLandmarkType.values[i],
        x: x,
        y: y,
        z: z,
        visibility: vis,
      ));
    }

    return PoseLandmarks(
      landmarks: lm,
      score: score,
    );
  }


  List<PoseLandmark> _transformLandmarks(
      PoseLandmarks landmarks,
      PoseDetection detection,
      int imageWidth,
      int imageHeight,
      ) {
    final usePad = (detection.width < 0.999 || detection.height < 0.999);
    final paddingFactor = usePad ? 1.8 : 1.0;

    final w = (detection.width * paddingFactor).clamp(0.1, 1.0);
    final h = (detection.height * paddingFactor).clamp(0.1, 1.0);
    final cx = detection.xCenter.clamp(0.0, 1.0);
    final cy = detection.yCenter.clamp(0.0, 1.0);

    final left = (cx - w / 2).clamp(0.0, 1.0);
    final top = (cy - h / 2).clamp(0.0, 1.0);

    return landmarks.landmarks.map((lm) {
      final nx = (left + lm.x * w).clamp(0.0, 1.0);
      final ny = (top + lm.y * h).clamp(0.0, 1.0);
      return PoseLandmark(
        type: lm.type,
        x: nx,
        y: ny,
        z: lm.z,
        visibility: lm.visibility,
      );
    }).toList();
  }

  List<List<List<List<double>>>> _imageToFloat32List(
      img.Image image,
      int width,
      int height,
      ) {
    final result = List.generate(
      1,
          (_) => List.generate(
        height,
            (_) => List.generate(
          width,
              (_) => List<double>.filled(3, 0.0),
        ),
      ),
    );

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        result[0][y][x][0] = pixel.r / 255.0;
        result[0][y][x][1] = pixel.g / 255.0;
        result[0][y][x][2] = pixel.b / 255.0;
      }
    }

    return result;
  }

  bool get isInitialized => _isInitialized;

  Future<void> dispose() async {
    _detectorInterpreter?.close();
    _landmarkInterpreter?.close();
    _detectorInterpreter = null;
    _landmarkInterpreter = null;
    _isInitialized = false;
  }
}

// ============================================================================
// DATA MODELS
// ============================================================================

/// Model complexity options
enum PoseModelComplexity {
  lite,   // 2.69 MB - Fastest
  full,   // 6.14 MB - Balanced
  heavy,  // 26.42 MB - Most accurate
}

/// Detection result from stage 1
class PoseDetection {
  final double xCenter; // Normalized 0-1
  final double yCenter; // Normalized 0-1
  final double width;   // Normalized 0-1
  final double height;  // Normalized 0-1
  final double score;
  final List<PoseKeypoint> keypoints;

  PoseDetection({
    required this.xCenter,
    required this.yCenter,
    required this.width,
    required this.height,
    required this.score,
    required this.keypoints,
  });
}

/// Keypoint from detector
class PoseKeypoint {
  final double x; // Normalized 0-1
  final double y; // Normalized 0-1
  final double score;

  PoseKeypoint({
    required this.x,
    required this.y,
    required this.score,
  });
}

/// Landmarks result from stage 2
class PoseLandmarks {
  final List<PoseLandmark> landmarks;
  final double score;

  PoseLandmarks({
    required this.landmarks,
    required this.score,
  });
}

/// Individual landmark point
class PoseLandmark {
  final PoseLandmarkType type;
  final double x; // Normalized 0-1
  final double y; // Normalized 0-1
  final double z; // Depth
  final double visibility;

  PoseLandmark({
    required this.type,
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });

  /// Convert to pixel coordinates
  Point toPixel(int imageWidth, int imageHeight) {
    return Point(
      (x * imageWidth).toInt(),
      (y * imageHeight).toInt(),
    );
  }
}

/// Complete pose detection result
class PoseDetectionResult {
  final List<PoseLandmark> landmarks;
  final PoseDetection detection;
  final int imageWidth;
  final int imageHeight;

  PoseDetectionResult({
    required this.landmarks,
    required this.detection,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// Get landmark by type
  PoseLandmark? getLandmark(PoseLandmarkType type) {
    try {
      return landmarks.firstWhere((l) => l.type == type);
    } catch (e) {
      return null;
    }
  }

  /// Check if pose is visible enough
  bool get isVisible {
    final avgVisibility = landmarks
        .map((l) => l.visibility)
        .reduce((a, b) => a + b) / landmarks.length;
    return avgVisibility > 0.5;
  }

  @override
  String toString() {
    final landmarksInfo = landmarks
        .map((l) => '${l.type.name}: (${l.x.toStringAsFixed(2)}, ${l.y.toStringAsFixed(2)}) vis=${l.visibility.toStringAsFixed(2)}')
        .join('\n');
    return 'PoseDetectionResult(\n'
        '  score=${detection.score.toStringAsFixed(3)},\n'
        '  landmarks=${landmarks.length},\n'
        '  visible=${isVisible},\n'
        '  coords:\n$landmarksInfo\n)';
  }
}

/// 33 pose landmarks following MediaPipe Pose convention
enum PoseLandmarkType {
  nose,                    // 0
  leftEyeInner,           // 1
  leftEye,                // 2
  leftEyeOuter,           // 3
  rightEyeInner,          // 4
  rightEye,               // 5
  rightEyeOuter,          // 6
  leftEar,                // 7
  rightEar,               // 8
  mouthLeft,              // 9
  mouthRight,             // 10
  leftShoulder,           // 11
  rightShoulder,          // 12
  leftElbow,              // 13
  rightElbow,             // 14
  leftWrist,              // 15
  rightWrist,             // 16
  leftPinky,              // 17
  rightPinky,             // 18
  leftIndex,              // 19
  rightIndex,             // 20
  leftThumb,              // 21
  rightThumb,             // 22
  leftHip,                // 23
  rightHip,               // 24
  leftKnee,               // 25
  rightKnee,              // 26
  leftAnkle,              // 27
  rightAnkle,             // 28
  leftHeel,               // 29
  rightHeel,              // 30
  leftFootIndex,          // 31
  rightFootIndex,         // 32
}

/// Simple point class
class Point {
  final int x;
  final int y;

  Point(this.x, this.y);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Helper to reshape a flat list into nested structure
List<List<List<double>>> _reshapeToTensor3D(
    List<double> flat,
    int dim1,
    int dim2,
    int dim3,
    ) {
  final result = List.generate(
    dim1,
        (_) => List.generate(
      dim2,
          (_) => List<double>.filled(dim3, 0.0),
    ),
  );

  int index = 0;
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      for (int k = 0; k < dim3; k++) {
        result[i][j][k] = flat[index++];
      }
    }
  }

  return result;
}

/// Helper to reshape a flat list into nested structure for 4D tensors
List<List<List<List<double>>>> _reshapeToTensor4D(
    List<double> flat,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
    ) {
  final result = List.generate(
    dim1,
        (_) => List.generate(
      dim2,
          (_) => List.generate(
        dim3,
            (_) => List<double>.filled(dim4, 0.0),
      ),
    ),
  );

  int index = 0;
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      for (int k = 0; k < dim3; k++) {
        for (int l = 0; l < dim4; l++) {
          result[i][j][k][l] = flat[index++];
        }
      }
    }
  }

  return result;
}