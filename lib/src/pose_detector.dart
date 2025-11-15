import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'types.dart';
import 'image_utils.dart';
import 'person_detector.dart';
import 'pose_landmark_model.dart';

class PoseDetector {
  final YoloV8PersonDetector _yolo = YoloV8PersonDetector();
  final PoseLandmarkModelRunner _lm = PoseLandmarkModelRunner();
  final PoseMode mode;
  final PoseLandmarkModel landmarkModel;
  final double detectorConf;
  final double detectorIou;
  final int maxDetections;
  final double minLandmarkScore;
  bool _isInitialized = false;
  img.Image? _canvasBuffer256;

  PoseDetector({
    this.mode = PoseMode.boxesAndLandmarks,
    this.landmarkModel = PoseLandmarkModel.heavy,
    this.detectorConf = 0.5,
    this.detectorIou = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
  });

  Future<void> initialize() async {
    if (_isInitialized) {
      await dispose();
    }
    await _lm.initialize(landmarkModel);
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

  Future<List<Pose>> detect(List<int> imageBytes) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }
    final img.Image? image = img.decodeImage(Uint8List.fromList(imageBytes));
    if (image == null) return <Pose>[];
    return detectOnImage(image);
  }

  Future<List<Pose>> detectOnImage(img.Image image) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }

    final List<YoloDetection> dets = await _yolo.detectOnImage(
      image,
      confThres: detectorConf,
      iouThres: detectorIou,
      topkPreNms: 100,
      maxDet: maxDetections,
      personOnly: true,
    );

    if (mode == PoseMode.boxes) {
      final List<Pose> out = <Pose>[];
      for (final YoloDetection d in dets) {
        out.add(
          Pose(
            boundingBox: BoundingBox(
              left: d.bboxXYXY[0],
              top: d.bboxXYXY[1],
              right: d.bboxXYXY[2],
              bottom: d.bboxXYXY[3],
            ),
            score: d.score,
            landmarks: const <PoseLandmark>[],
            imageWidth: image.width,
            imageHeight: image.height,
          ),
        );
      }
      return out;
    }

    final List<Pose> results = <Pose>[];
    for (final YoloDetection d in dets) {
      final int x1 = d.bboxXYXY[0].clamp(0.0, image.width.toDouble()).toInt();
      final int y1 = d.bboxXYXY[1].clamp(0.0, image.height.toDouble()).toInt();
      final int x2 = d.bboxXYXY[2].clamp(0.0, image.width.toDouble()).toInt();
      final int y2 = d.bboxXYXY[3].clamp(0.0, image.height.toDouble()).toInt();
      final int cw = (x2 - x1).clamp(1, image.width);
      final int ch = (y2 - y1).clamp(1, image.height);

      final img.Image crop = img.copyCrop(image, x: x1, y: y1, width: cw, height: ch);
      final List<double> ratio = <double>[];
      final List<int> dwdh = <int>[];
      _canvasBuffer256 ??= img.Image(width: 256, height: 256);
      final img.Image letter = ImageUtils.letterbox256(
        crop,
        ratio,
        dwdh,
        reuseCanvas: _canvasBuffer256
      );
      final double r = ratio.first;
      final int dw = dwdh[0];
      final int dh = dwdh[1];

      final PoseLandmarks lms = await _lm.run(letter);
      if (lms.score < minLandmarkScore) continue;

      final List<PoseLandmark> pts = <PoseLandmark>[];
      for (final PoseLandmark lm in lms.landmarks) {
        final double xp = lm.x * 256.0;
        final double yp = lm.y * 256.0;
        final double xContent = (xp - dw) / r;
        final double yContent = (yp - dh) / r;
        final double xOrig = (x1.toDouble() + xContent)
            .clamp(0.0, image.width.toDouble());
        final double yOrig = (y1.toDouble() + yContent)
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
        Pose(
          boundingBox: BoundingBox(
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
