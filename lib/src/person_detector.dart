import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'image_utils.dart';

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

  List<List<List<List<double>>>>? _inputBuffer;
  Map<int, Object>? _outputBuffers;
  img.Image? _canvasBuffer;

  Future<void> initialize() async {
    const assetPath = 'packages/pose_detection_tflite/assets/models/yolov8n_float32.tflite';
    if (_isInitialized) await dispose();
    _interpreter = await Interpreter.fromAsset(assetPath);
    _interpreter!.allocateTensors();

    final inTensor = _interpreter!.getInputTensor(0);
    final inShape = inTensor.shape;
    _inH = inShape[1];
    _inW = inShape[2];

    _outShapes.clear();
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
    _inputBuffer = null;
    _outputBuffers = null;
    _canvasBuffer = null;
    _isInitialized = false;
  }

  static double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  static List<int> _argSortDesc(List<double> a) {
    final idx = List<int>.generate(a.length, (i) => i);
    idx.sort((i, j) => a[j].compareTo(a[i]));
    return idx;
  }

  static List<int> _nms(
    List<List<double>> boxes,
    List<double> scores, {
    double iouThres = 0.45,
    int maxDet = 100,
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

    double area(List<double> b) =>
        math.max(0.0, b[2] - b[0]) * math.max(0.0, b[3] - b[1]);

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
    return raw.map<List<double>>((e) => (e as List).map((v) => (v as num)
        .toDouble()).toList()).toList();
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
    _canvasBuffer ??= img.Image(width: _inW, height: _inH);
    final letter = ImageUtils.letterbox(image, _inW, _inH, ratio, dwdh, reuseCanvas: _canvasBuffer);
    final r = ratio.first;
    final dw = dwdh[0], dh = dwdh[1];

    if (_inputBuffer == null) {
      _inputBuffer = List.generate(
        1,
            (_) => List.generate(
          _inH,
              (_) => List.generate(
            _inW,
                (_) => List<double>.filled(3, 0.0),
            growable: false,
          ),
          growable: false,
        ),
        growable: false,
      );
    }

    final input = _inputBuffer!;
    for (int y = 0; y < _inH; y++) {
      for (int x = 0; x < _inW; x++) {
        final px = letter.getPixel(x, y);
        input[0][y][x][0] = px.r / 255.0;
        input[0][y][x][1] = px.g / 255.0;
        input[0][y][x][2] = px.b / 255.0;
      }
    }

    if (_outputBuffers == null) {
      _outputBuffers = <int, Object>{};
      for (int i = 0; i < _outShapes.length; i++) {
        final shape = _outShapes[i];
        Object buf;
        if (shape.length == 3) {
          buf = List.generate(
            shape[0],
                (_) => List.generate(
              shape[1],
                  (_) => List<double>.filled(shape[2], 0.0),
              growable: false,
            ),
            growable: false,
          );
        } else if (shape.length == 2) {
          buf = List.generate(
            shape[0],
                (_) => List<double>.filled(shape[1], 0.0),
            growable: false,
          );
        } else {
          buf = List.filled(shape.reduce((a, b) => a * b), 0.0);
        }
        _outputBuffers![i] = buf;
      }
    }

    final outputs = _outputBuffers!;

    _interpreter!.runForMultipleInputs([input], outputs);

    final decoded = _decodeAnyYoloOutputs(outputs.values.toList());

    final scores = <double>[];
    final clsIds = <int>[];
    final xywhs = <List<double>>[];

    for (final row in decoded) {
      final C = row['C'] as int;
      final xywh = (row['xywh'] as List).map((v) =>
          (v as num).toDouble()).toList();
      final rest = (row['rest'] as List).map((v) =>
          (v as num).toDouble()).toList();

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
      boxes.add(ImageUtils.scaleFromLetterbox(b, r, dw, dh));
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
        bboxXYXY: boxes[i],
      ));
    }
    return out;
  }
}
