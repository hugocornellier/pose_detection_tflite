import 'dart:io';
import 'dart:math' as math;
import 'dart:ffi' as ffi;

import 'package:image/image.dart' as img;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'package:path/path.dart' as p;

class PoseDetector {
  Interpreter? _detectorInterpreter;
  Interpreter? _landmarkInterpreter;

  bool _isInitialized = false;
  late PoseModelComplexity _complexity;

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
  }

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

  Future<void> initialize({
    PoseModelComplexity complexity = PoseModelComplexity.heavy,
  }) async {
    if (_isInitialized) {
      await dispose();
    }

    await _ensureTFLiteLoaded();
    _complexity = complexity;

    _detectorInterpreter = await _loadModelFromAssets(
      'assets/models/pose_detection.tflite',
    );
    _landmarkInterpreter = await _loadModelFromAssets(
      _getLandmarkModelPath(complexity),
    );

    _detectorInterpreter!.resizeInputTensor(0, [1, 224, 224, 3]);
    _detectorInterpreter!.allocateTensors();

    _landmarkInterpreter!.resizeInputTensor(0, [1, 256, 256, 3]);
    _landmarkInterpreter!.allocateTensors();

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

  Future<PoseDetectionResult?> detectPose(File imageFile) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }

    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes);
    if (image == null) return null;

    return detectPoseFromImage(image);
  }

  Future<PoseDetectionResult?> detectPoseFromImage(img.Image image) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }

    final W = image.width;
    final H = image.height;

    final s = math.min(256.0 / W, 256.0 / H);
    final wScaled = (W * s).round();
    final hScaled = (H * s).round();

    final leftPad = ((256 - wScaled) / 2).floor();
    final topPad = ((256 - hScaled) / 2).floor();

    final resized = img.copyResize(image, width: wScaled, height: hScaled);
    final canvas = img.Image(width: 256, height: 256);
    img.compositeImage(canvas, resized, dstX: leftPad, dstY: topPad);

    final landmarks = await _runLandmarkDetection(canvas);

    if (landmarks.score < 0.5) {
      return null;
    }

    final out = <PoseLandmark>[];
    for (final lm in landmarks.landmarks) {
      final xp = lm.x * 256.0;
      final yp = lm.y * 256.0;

      final xContent = xp - leftPad;
      final yContent = yp - topPad;

      final xOrigPx = xContent / s;
      final yOrigPx = yContent / s;

      final xPx = xOrigPx.clamp(0.0, W.toDouble());
      final yPx = yOrigPx.clamp(0.0, H.toDouble());

      out.add(PoseLandmark(
        type: lm.type,
        x: xPx,
        y: yPx,
        z: lm.z,
        visibility: lm.visibility,
      ));
    }


    return PoseDetectionResult(
      landmarks: out,
      detection: PoseDetection(
        xCenter: 0.5,
        yCenter: 0.5,
        width: 1.0,
        height: 1.0,
        score: 1.0,
        keypoints: const [],
      ),
      imageWidth: W,
      imageHeight: H,
    );
  }

  List<List<List<List<double>>>>? _nhwc256Cache;

  List<List<List<List<double>>>> _imageToNHWC4D(
    img.Image image,
    int width,
    int height,
  ) {
    _nhwc256Cache ??= List.generate(
      1,
          (_) => List.generate(
        height,
            (_) => List.generate(
          width,
              (_) => List<double>.filled(3, 0.0),
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );

    final out = _nhwc256Cache!;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final px = image.getPixel(x, y);
        out[0][y][x][0] = px.r / 255.0;
        out[0][y][x][1] = px.g / 255.0;
        out[0][y][x][2] = px.b / 255.0;
      }
    }
    return out;
  }

  Future<PoseLandmarks> _runLandmarkDetection(img.Image roiImage) async {
    final input4d = _imageToNHWC4D(roiImage, 256, 256);

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

    _landmarkInterpreter!.runForMultipleInputs(
      [input4d],
      {
        0: outputLandmarks,
        1: outputScore,
        2: outputMask,
        3: outputHeatmap,
        4: outputWorld,
      },
    );

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

  bool get isInitialized => _isInitialized;

  Future<void> dispose() async {
    _detectorInterpreter?.close();
    _landmarkInterpreter?.close();
    _detectorInterpreter = null;
    _landmarkInterpreter = null;
    _isInitialized = false;
  }
}

enum PoseModelComplexity {
  lite,   // 2.69 MB - Fastest
  full,   // 6.14 MB - Balanced
  heavy,  // 26.42 MB - Most accurate
}

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

class PoseLandmarks {
  final List<PoseLandmark> landmarks;
  final double score;

  PoseLandmarks({
    required this.landmarks,
    required this.score,
  });
}

class PoseLandmark {
  final PoseLandmarkType type;
  final double x; // Pixels
  final double y; // Pixels
  final double z;
  final double visibility;

  PoseLandmark({
    required this.type,
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });

  double xNorm(int imageWidth) => (x / imageWidth).clamp(0.0, 1.0);
  double yNorm(int imageHeight) => (y / imageHeight).clamp(0.0, 1.0);

  Point toPixel(int imageWidth, int imageHeight) {
    return Point(x.toInt(), y.toInt());
  }
}


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

  PoseLandmark? getLandmark(PoseLandmarkType type) {
    try {
      return landmarks.firstWhere((l) => l.type == type);
    } catch (e) {
      return null;
    }
  }

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

class Point {
  final int x;
  final int y;

  Point(this.x, this.y);
}

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