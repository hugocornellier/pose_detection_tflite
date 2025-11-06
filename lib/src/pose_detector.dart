import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'types.dart';
import 'image_utils.dart';
import 'person_detector.dart';
import 'pose_landmark_model.dart';

class PoseDetector {
  bool _isInitialized = false;
  late PoseOptions _opts;

  final YoloV8PersonDetector _yolo = YoloV8PersonDetector();
  final PoseLandmarkModelRunner _lm = PoseLandmarkModelRunner();
  img.Image? _canvasBuffer256;

  Future<void> initialize({PoseOptions options = const PoseOptions()}) async {
    if (_isInitialized) {
      await dispose();
    }
    _opts = options;
    await _lm.initialize(_opts.landmarkModel);
    await _yolo.initialize();
    _isInitialized = true;
  }

  bool get isInitialized => _isInitialized;

  Future<void> dispose() async {
    await _yolo.dispose();
    await _lm.dispose();
    _canvasBuffer256 = null;
    _isInitialized = false;
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
      final cw = (x2 - x1).clamp(1, image.width);
      final ch = (y2 - y1).clamp(1, image.height);

      final crop = img.copyCrop(image, x: x1, y: y1, width: cw, height: ch);
      final ratio = <double>[];
      final dwdh = <int>[];
      _canvasBuffer256 ??= img.Image(width: 256, height: 256);
      final letter = ImageUtils.letterbox256(crop, ratio, dwdh, reuseCanvas: _canvasBuffer256);
      final r = ratio.first;
      final dw = dwdh[0];
      final dh = dwdh[1];

      final lms = _lm.run(letter);
      if (lms.score < _opts.minLandmarkScore) continue;

      final pts = <PoseLandmark>[];
      for (final lm in lms.landmarks) {
        final xp = lm.x * 256.0;
        final yp = lm.y * 256.0;
        final xContent = (xp - dw) / r;
        final yContent = (yp - dh) / r;
        final xOrig = (x1.toDouble() + xContent)
            .clamp(0.0, image.width.toDouble());
        final yOrig = (y1.toDouble() + yContent)
            .clamp(0.0, image.height.toDouble());
        pts.add(
          PoseLandmark(
            type: lm.type,
            x: xOrig,
            y: yOrig,
            z: lm.z,
            visibility: lm.visibility,
          ),
        );
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
}
