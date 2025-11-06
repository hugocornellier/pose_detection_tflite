import 'dart:math' as math;
import 'package:image/image.dart' as img;

class ImageUtils {
  static img.Image letterbox(
    img.Image src,
    int tw,
    int th,
    List<double> ratioOut,
    List<int> dwdhOut, {
    img.Image? reuseCanvas,
  }) {
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
      interpolation: img.Interpolation.linear,
    );
    final canvas = reuseCanvas ?? img.Image(width: tw, height: th);
    if (canvas.width != tw || canvas.height != th) {
      throw ArgumentError(
          'Reuse canvas dimensions (${canvas.width}x${canvas.height}) '
              'do not match target dimensions (${tw}x${th})'
      );
    }
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
    img.compositeImage(canvas, resized, dstX: dw, dstY: dh);

    ratioOut..clear()..add(r);
    dwdhOut..clear()..addAll([dw, dh]);
    return canvas;
  }

  static img.Image letterbox256(
    img.Image src,
    List<double> ratioOut,
    List<int> dwdhOut, {
    img.Image? reuseCanvas,
  }) {
    return letterbox(src, 256, 256, ratioOut, dwdhOut, reuseCanvas: reuseCanvas);
  }

  static List<double> scaleFromLetterbox(
    List<double> xyxy,
    double ratio,
    int dw,
    int dh,
  ) {
    final x1 = (xyxy[0] - dw) / ratio;
    final y1 = (xyxy[1] - dh) / ratio;
    final x2 = (xyxy[2] - dw) / ratio;
    final y2 = (xyxy[3] - dh) / ratio;
    return [x1, y1, x2, y2];
  }

  static List<List<List<List<double>>>> imageToNHWC4D(
    img.Image image,
    int width,
    int height, {
    List<List<List<List<double>>>>? reuse,
  }) {
    final out = reuse ??
        List.generate(
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

  static List<List<List<List<double>>>> reshapeToTensor4D(
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
}
