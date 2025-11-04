import 'dart:io';
import 'dart:math' as math;
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'package:path/path.dart' as p;

class PoseDetector {
  Interpreter? _detectorInterpreter;
  Interpreter? _landmarkInterpreter;

  bool _isInitialized = false;
  late PoseOptions _opts;
  final YoloV8PersonDetector _yolo = YoloV8PersonDetector();
  static ffi.DynamicLibrary? _tfliteLib;

  img.Image _letterbox256(
    img.Image src,
    List<double> ratioOut,
    List<int> dwdhOut
  ) {
    final w = src.width;
    final h = src.height;
    final r = math.min(256 / h, 256 / w);
    final nw = (w * r).round();
    final nh = (h * r).round();
    final dw = (256 - nw) ~/ 2;
    final dh = (256 - nh) ~/ 2;

    final resized = img.copyResize(
      src,
      width: nw,
      height: nh,
      interpolation: img.Interpolation.linear
    );
    final canvas = img.Image(width: 256, height: 256);
    img.compositeImage(canvas, resized, dstX: dw, dstY: dh);

    ratioOut..clear()..add(r);
    dwdhOut..clear()..addAll([dw, dh]);
    return canvas;
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

  Future<void> initialize({PoseOptions options = const PoseOptions()}) async {
    if (_isInitialized) {
      await dispose();
    }

    await _ensureTFLiteLoaded();
    _opts = options;

    _landmarkInterpreter = await _loadModelFromAssets(
      _getLandmarkModelPath(_opts.landmarkModel),
    );
    _landmarkInterpreter!.resizeInputTensor(0, [1, 256, 256, 3]);
    _landmarkInterpreter!.allocateTensors();

    await _yolo.initialize();

    _isInitialized = true;
  }

  String _getLandmarkModelPath(PoseLandmarkModel model) {
    switch (model) {
      case PoseLandmarkModel.lite:
        return 'packages/pose_detection_tflite/assets/models/pose_landmark_lite.tflite';
      case PoseLandmarkModel.full:
        return 'packages/pose_detection_tflite/assets/models/pose_landmark_full.tflite';
      case PoseLandmarkModel.heavy:
        return 'packages/pose_detection_tflite/assets/models/pose_landmark_heavy.tflite';
    }
  }

  Future<Interpreter> _loadModelFromAssets(String assetPath) async {
    return await Interpreter.fromAsset(assetPath);
  }

  Future<List<PoseResult>> detect(List<int> imageBytes) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }
    final image = img.decodeImage(Uint8List.fromList(imageBytes));
    if (image == null) return <PoseResult>[];
    return detectOnImage(image);
  }

  Future<List<PoseResult>> detectOnImage(img.Image image) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }

    final dets = _yolo.detectOnImage(
      image,
      confThres: _opts.detectorConf,
      iouThres: _opts.detectorIou,
      topkPreNms: 100,
      maxDet: _opts.maxDetections,
      personOnly: true,
    );

    if (_opts.mode == PoseMode.boxes) {
      final out = <PoseResult>[];
      for (final d in dets) {
        out.add(
          PoseResult(
            bboxPx: RectPx(
              left: d.bboxXYXY[0],
              top: d.bboxXYXY[1],
              right: d.bboxXYXY[2],
              bottom: d.bboxXYXY[3],
            ),
            score: d.score,
            landmarks: null,
            imageWidth: image.width,
            imageHeight: image.height,
          ),
        );
      }
      return out;
    }

    final results = <PoseResult>[];
    for (final d in dets) {
      final x1 = d.bboxXYXY[0].clamp(0.0, image.width.toDouble()).toInt();
      final y1 = d.bboxXYXY[1].clamp(0.0, image.height.toDouble()).toInt();
      final x2 = d.bboxXYXY[2].clamp(0.0, image.width.toDouble()).toInt();
      final y2 = d.bboxXYXY[3].clamp(0.0, image.height.toDouble()).toInt();
      final cw = math.max(1, x2 - x1);
      final ch = math.max(1, y2 - y1);

      final crop = img.copyCrop(image, x: x1, y: y1, width: cw, height: ch);
      final ratio = <double>[];
      final dwdh = <int>[];
      final letter = _letterbox256(crop, ratio, dwdh);
      final r = ratio.first;
      final dw = dwdh[0];
      final dh = dwdh[1];

      final lms = await _runLandmarkDetection(letter);
      if (lms.score < _opts.minLandmarkScore) continue;

      final pts = <PoseLandmark>[];
      for (final lm in lms.landmarks) {
        final xp = lm.x * 256.0;
        final yp = lm.y * 256.0;
        final xContent = (xp - dw) / r;
        final yContent = (yp - dh) / r;
        final xOrig = (x1.toDouble() + xContent).clamp(
          0.0,
          image.width.toDouble()
        );
        final yOrig = (y1.toDouble() + yContent).clamp(
          0.0,
          image.height.toDouble()
        );
        pts.add(PoseLandmark(
          type: lm.type,
          x: xOrig,
          y: yOrig,
          z: lm.z,
          visibility: lm.visibility,
        ));
      }

      results.add(
        PoseResult(
          bboxPx: RectPx(
            left: d.bboxXYXY[0],
            top: d.bboxXYXY[1],
            right: d.bboxXYXY[2],
            bottom: d.bboxXYXY[3],
          ),
          score: d.score,
          landmarks: pts,
          imageWidth: image.width,
          imageHeight: image.height,
        ),
      );
    }

    return results;
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
    double _clamp01(double v) => v.isNaN
        ? 0.0
        : v < 0.0
          ? 0.0
          : (v > 1.0 ? 1.0 : v);

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
    await _yolo.dispose();
    _detectorInterpreter = null;
    _landmarkInterpreter = null;
    _isInitialized = false;
  }

}

enum PoseLandmarkModel {
  lite,
  full,
  heavy,
}

enum PoseMode {
  boxes,
  boxesAndLandmarks,
}

class PoseOptions {
  final PoseMode mode;
  final PoseLandmarkModel landmarkModel;
  final double detectorConf;
  final double detectorIou;
  final int maxDetections;
  final double minLandmarkScore;

  const PoseOptions({
    this.mode = PoseMode.boxesAndLandmarks,
    this.landmarkModel = PoseLandmarkModel.heavy,
    this.detectorConf = 0.5,
    this.detectorIou = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
  });
}

class PoseDetection {
  final double xCenter;
  final double yCenter;
  final double width;
  final double height;
  final double score;
  final List<PoseKeypoint> keypoints;

  const PoseDetection({
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

class RectPx {
  final double left;
  final double top;
  final double right;
  final double bottom;

  const RectPx({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom
  });
}

class PoseResult {
  final RectPx bboxPx;
  final double score;
  final List<PoseLandmark>? landmarks;
  final int imageWidth;
  final int imageHeight;

  const PoseResult({
    required this.bboxPx,
    required this.score,
    required this.landmarks,
    required this.imageWidth,
    required this.imageHeight,
  });

  PoseLandmark? getLandmark(PoseLandmarkType type) {
    if (landmarks == null) return null;
    try {
      return landmarks!.firstWhere((l) => l.type == type);
    } catch (_) {
      return null;
    }
  }

  bool get hasLandmarks => landmarks != null && landmarks!.isNotEmpty;

  @override
  String toString() {
    final lm = landmarks ?? const <PoseLandmark>[];
    final landmarksInfo = lm
        .map((l) => '${l.type.name}: (${l.x.toStringAsFixed(2)}, ${l.y.toStringAsFixed(2)}) vis=${l.visibility.toStringAsFixed(2)}')
        .join('\n');
    return 'PoseResult(\n'
        '  score=${score.toStringAsFixed(3)},\n'
        '  landmarks=${lm.length},\n'
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

class YoloDetection {
  final int cls;
  final double score;
  final List<double> bboxXYXY;

  YoloDetection({
    required this.cls,
    required this.score,
    required this.bboxXYXY,
  });
}

class YoloV8PersonDetector {
  static const int cocoPersonClassId = 0;

  Interpreter? _interpreter;
  bool _isInitialized = false;

  late int _inW;
  late int _inH;

  final _outShapes = <List<int>>[];

  Future<void> initialize() async {
    const assetPath = 'packages/pose_detection_tflite/assets/models/yolov8n_float32.tflite';

    if (_isInitialized) await dispose();
    _interpreter = await Interpreter.fromAsset(assetPath);
    _interpreter!.allocateTensors();

    final inTensor = _interpreter!.getInputTensor(0);
    final inShape = inTensor.shape;
    _inH = inShape[1];
    _inW = inShape[2];

    _outShapes
      ..clear();
    final outs = _interpreter!.getOutputTensors();
    for (final t in outs) {
      _outShapes.add(List<int>.from(t.shape));
    }

    _isInitialized = true;
  }

  bool get isInitialized => _isInitialized;

  Future<void> dispose() async {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }

  static double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  static img.Image _letterbox(
    img.Image src,
    int tw,
    int th,
    List<double> ratioOut,
    List<int> dwdhOut
  ) {
    final w = src.width;
    final h = src.height;
    final r = math.min(th / h, tw / w);
    final nw = (w * r).round();
    final nh = (h * r).round();
    final dw = (tw - nw) ~/ 2;
    final dh = (th - nh) ~/ 2;

    final resized = img.copyResize(
      src,
      width: nw,
      height: nh,
      interpolation: img.Interpolation.linear
    );
    final canvas = img.Image(width: tw, height: th);
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
    img.compositeImage(canvas, resized, dstX: dw, dstY: dh);

    ratioOut..clear()..add(r);
    dwdhOut..clear()..addAll([dw, dh]);
    return canvas;
  }

  static List<double> _scaleFromLetterbox(
    List<double> xyxy,
    double ratio,
    int dw,
    int dh
  ) {
    final x1 = (xyxy[0] - dw) / ratio;
    final y1 = (xyxy[1] - dh) / ratio;
    final x2 = (xyxy[2] - dw) / ratio;
    final y2 = (xyxy[3] - dh) / ratio;
    return [x1, y1, x2, y2];
  }

  static List<int> _argSortDesc(List<double> a) {
    final idx = List<int>.generate(a.length, (i) => i);
    idx.sort((i, j) => a[j].compareTo(a[i]));
    return idx;
  }

  static List<int> _nms(
    List<List<double>> boxes,
    List<double> scores, {
    double iouThres = 0.45,
    int maxDet = 100
  }) {
    if (boxes.isEmpty) return <int>[];
    final order = _argSortDesc(scores);
    final keep = <int>[];

    double interArea(List<double> a, List<double> b) {
      final xx1 = math.max(a[0], b[0]);
      final yy1 = math.max(a[1], b[1]);
      final xx2 = math.min(a[2], b[2]);
      final yy2 = math.min(a[3], b[3]);
      final w = math.max(0.0, xx2 - xx1);
      final h = math.max(0.0, yy2 - yy1);
      return w * h;
    }

    double area(List<double> b) => math.max(0.0, b[2] - b[0]) * math.max(0.0, b[3] - b[1]);
    final areas = boxes.map(area).toList();

    final suppressed = List<bool>.filled(order.length, false);
    for (int m = 0; m < order.length; m++) {
      if (suppressed[m]) continue;
      final i = order[m];
      keep.add(i);
      if (keep.length >= maxDet) break;
      for (int n = m + 1; n < order.length; n++) {
        if (suppressed[n]) continue;
        final j = order[n];
        final inter = interArea(boxes[i], boxes[j]);
        final u = areas[i] + areas[j] - inter + 1e-7;
        final iou = inter / u;
        if (iou > iouThres) suppressed[n] = true;
      }
    }
    return keep;
  }

  static List<List<double>> _transpose2D(List<List<double>> a) {
    if (a.isEmpty) return <List<double>>[];
    final rows = a.length, cols = a[0].length;
    final out = List.generate(cols, (_) => List<double>.filled(rows, 0.0));
    for (int r = 0; r < rows; r++) {
      final row = a[r];
      for (int c = 0; c < cols; c++) {
        out[c][r] = row[c];
      }
    }
    return out;
  }

  static List<List<double>> _concat0(List<List<List<double>>> parts) {
    final out = <List<double>>[];
    for (final p in parts) {
      out.addAll(p);
    }
    return out;
  }

  static List<List<double>> _ensure2D(List<dynamic> raw) {
    return raw.map<List<double>>((e) => (e as List).map((v) => (v as num).toDouble()).toList()).toList();
  }

  List<Map<String, dynamic>> _decodeAnyYoloOutputs(List<dynamic> outputs) {
    final parts = <List<List<double>>>[];
    for (final raw in outputs) {
      final t3d = raw as List;
      if (t3d.length != 1) throw StateError('Unexpected YOLO output rank');
      final out2d = _ensure2D(t3d[0]);
      if (out2d.isEmpty) continue;
      final rows = out2d.length;
      final cols = out2d[0].length;
      if (rows < cols && (rows == 84 || rows == 85)) {
        parts.add(_transpose2D(out2d));
      } else {
        parts.add(out2d);
      }
    }
    final out = _concat0(parts);
    if (out.isEmpty || out[0].length < 84) throw StateError('Expected channels >=84');
    final channels = out[0].length;
    return out
        .map((row) => {
      'xywh': row.sublist(0, 4),
      'rest': row.sublist(4),
      'C': channels,
    })
        .toList();
  }

  static List<double> _xywhToXyxy(List<double> xywh) {
    final cx = xywh[0], cy = xywh[1], w = xywh[2], h = xywh[3];
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
  }

  static double _median(List<double> a) {
    if (a.isEmpty) return double.nan;
    final b = List<double>.from(a)..sort();
    final n = b.length;
    if (n.isOdd) return b[n ~/ 2];
    return 0.5 * (b[n ~/ 2 - 1] + b[n ~/ 2]);
  }

  List<YoloDetection> detectOnImage(
    img.Image image, {
    double confThres = 0.35,
    double iouThres = 0.4,
    int topkPreNms = 100,
    int maxDet = 10,
    bool personOnly = true,
  }) {
    if (!_isInitialized || _interpreter == null) {
      throw StateError('YoloV8PersonDetector not initialized.');
    }

    final ratio = <double>[];
    final dwdh = <int>[];
    final letter = _letterbox(image, _inW, _inH, ratio, dwdh);
    final r = ratio.first;
    final dw = dwdh[0], dh = dwdh[1];

    final input = List.generate(
      1,
      (_) => List.generate(
        _inH,
        (_) => List.generate(
          _inW,
          (_) => List<double>.filled(3, 0.0),
          growable: false
        ),
        growable: false,
      ),
      growable: false,
    );

    for (int y = 0; y < _inH; y++) {
      for (int x = 0; x < _inW; x++) {
        final px = letter.getPixel(x, y);
        input[0][y][x][0] = px.r / 255.0;
        input[0][y][x][1] = px.g / 255.0;
        input[0][y][x][2] = px.b / 255.0;
      }
    }

    final outputs = <int, Object>{};
    for (int i = 0; i < _outShapes.length; i++) {
      final shape = _outShapes[i];
      Object buf;
      if (shape.length == 3) {
        buf = List.generate(
          shape[0],
          (_) => List.generate(
            shape[1],
            (_) => List<double>.filled(shape[2], 0.0),
            growable: false
          ),
          growable: false,
        );
      } else if (shape.length == 2) {
        buf = List.generate(
          shape[0],
          (_) => List<double>.filled(shape[1], 0.0),
          growable: false
        );
      } else {
        buf = List.filled(shape.reduce((a, b) => a * b), 0.0);
      }
      outputs[i] = buf;
    }

    _interpreter!.runForMultipleInputs([input], outputs);

    final decoded = _decodeAnyYoloOutputs(outputs.values.toList());

    final scores = <double>[];
    final clsIds = <int>[];
    final xywhs = <List<double>>[];

    for (final row in decoded) {
      final C = row['C'] as int;
      final xywh = (row['xywh'] as List).map((v) => (v as num).toDouble()).toList();
      final rest = (row['rest'] as List).map((v) => (v as num).toDouble()).toList();

      if (C == 84) {
        int argMax = 0;
        double best = -1e9;
        for (int i = 0; i < rest.length; i++) {
          final s = _sigmoid(rest[i]);
          if (s > best) {
            best = s;
            argMax = i;
          }
        }
        scores.add(best);
        clsIds.add(argMax);
        xywhs.add(xywh);
      } else {
        final obj = _sigmoid(rest[0]);
        final clsLogits = rest.sublist(1, 81);
        int argMax = 0;
        double best = -1e9;
        for (int i = 0; i < clsLogits.length; i++) {
          final s = _sigmoid(clsLogits[i]);
          if (s > best) {
            best = s;
            argMax = i;
          }
        }
        scores.add(obj * best);
        clsIds.add(argMax);
        xywhs.add(xywh);
      }
    }

    final keep0 = <int>[];
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] >= confThres) keep0.add(i);
    }
    if (keep0.isEmpty) return <YoloDetection>[];

    final keptXywh = [for (final i in keep0) xywhs[i]];
    final keptCls = [for (final i in keep0) clsIds[i]];
    final keptScore = [for (final i in keep0) scores[i]];

    if (keptXywh.isNotEmpty && _median([for (final v in keptXywh) v[2]]) <= 2.0) {
      for (final v in keptXywh) {
        v[0] *= _inW.toDouble();
        v[1] *= _inH.toDouble();
        v[2] *= _inW.toDouble();
        v[3] *= _inH.toDouble();
      }
    }

    final boxesLtr = [for (final v in keptXywh) _xywhToXyxy(v)];
    final boxes = <List<double>>[];
    for (final b in boxesLtr) {
      boxes.add(_scaleFromLetterbox(b, r, dw, dh));
    }
    final iw = image.width.toDouble();
    final ih = image.height.toDouble();
    for (final b in boxes) {
      b[0] = b[0].clamp(0.0, iw);
      b[2] = b[2].clamp(0.0, iw);
      b[1] = b[1].clamp(0.0, ih);
      b[3] = b[3].clamp(0.0, ih);
    }

    if (topkPreNms > 0 && keptScore.length > topkPreNms) {
      final ord = _argSortDesc(keptScore).take(topkPreNms).toList();
      final _boxes = <List<double>>[];
      final _scores = <double>[];
      final _cls = <int>[];
      for (final i in ord) {
        _boxes.add(boxes[i]);
        _scores.add(keptScore[i]);
        _cls.add(keptCls[i]);
      }
      boxes
        ..clear()
        ..addAll(_boxes);
      keptScore
        ..clear()
        ..addAll(_scores);
      keptCls
        ..clear()
        ..addAll(_cls);
    }

    if (personOnly) {
      final fBoxes = <List<double>>[];
      final fScores = <double>[];
      final fCls = <int>[];
      for (int i = 0; i < keptCls.length; i++) {
        if (keptCls[i] == cocoPersonClassId) {
          fBoxes.add(boxes[i]);
          fScores.add(keptScore[i]);
          fCls.add(keptCls[i]);
        }
      }
      boxes
        ..clear()
        ..addAll(fBoxes);
      keptScore
        ..clear()
        ..addAll(fScores);
      keptCls
        ..clear()
        ..addAll(fCls);
    }

    final keep = _nms(boxes, keptScore, iouThres: iouThres, maxDet: maxDet);

    final out = <YoloDetection>[];
    for (final i in keep) {
      out.add(YoloDetection(
        cls: keptCls[i],
        score: keptScore[i],
        bboxXYXY: boxes[i]
      ));
    }
    return out;
  }
}
