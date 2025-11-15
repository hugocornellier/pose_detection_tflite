enum PoseLandmarkModel { lite, full, heavy }
enum PoseMode { boxes, boxesAndLandmarks }

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
  final double x;
  final double y;
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

enum PoseLandmarkType {
  nose,
  leftEyeInner,
  leftEye,
  leftEyeOuter,
  rightEyeInner,
  rightEye,
  rightEyeOuter,
  leftEar,
  rightEar,
  mouthLeft,
  mouthRight,
  leftShoulder,
  rightShoulder,
  leftElbow,
  rightElbow,
  leftWrist,
  rightWrist,
  leftPinky,
  rightPinky,
  leftIndex,
  rightIndex,
  leftThumb,
  rightThumb,
  leftHip,
  rightHip,
  leftKnee,
  rightKnee,
  leftAnkle,
  rightAnkle,
  leftHeel,
  rightHeel,
  leftFootIndex,
  rightFootIndex,
}

class Point {
  final int x;
  final int y;

  Point(this.x, this.y);
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
    required this.bottom,
  });
}

class Pose {
  final RectPx bboxPx;
  final double score;
  final List<PoseLandmark> landmarks;
  final int imageWidth;
  final int imageHeight;

  const Pose({
    required this.bboxPx,
    required this.score,
    required this.landmarks,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// Gets a specific landmark by type, or null if not found
  PoseLandmark? getLandmark(PoseLandmarkType type) {
    try {
      return landmarks.firstWhere((l) => l.type == type);
    } catch (_) {
      return null;
    }
  }

  /// Returns true if this pose has landmarks
  bool get hasLandmarks => landmarks.isNotEmpty;

  @override
  String toString() {
    final String landmarksInfo = landmarks
        .map((l) => '${l.type.name}: (${l.x.toStringAsFixed(2)}, ${l.y.toStringAsFixed(2)}) vis=${l.visibility.toStringAsFixed(2)}')
        .join('\n');
    return 'Pose(\n'
        '  score=${score.toStringAsFixed(3)},\n'
        '  landmarks=${landmarks.length},\n'
        '  coords:\n$landmarksInfo\n)';
  }
}