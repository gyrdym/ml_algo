import 'package:linalg/vector.dart';

class InterceptPreprocessor {
  final double _interceptScale;

  const InterceptPreprocessor({double interceptScale = 1.0}) : _interceptScale = interceptScale;

  List<Float32x4Vector> addIntercept(List<Float32x4Vector> points) {
    if (_interceptScale == 0.0) {
      return points;
    }

    final _points = new List<Float32x4Vector>(points.length);
    for (int i = 0; i < points.length; i++) {
      _points[i] = new Float32x4Vector.from(
        points[i].toList(growable: true)
          ..insert(0, 1.0 * _interceptScale)
      );
    }
    return _points;
  }
}