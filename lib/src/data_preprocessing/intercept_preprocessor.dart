import 'dart:typed_data';

import 'package:linalg/linalg.dart';

class InterceptPreprocessor {
  final double _interceptScale;

  const InterceptPreprocessor({double interceptScale = 1.0}) : _interceptScale = interceptScale;

  List<Vector<Float32x4>> addIntercept(List<Vector<Float32x4>> points) {
    if (_interceptScale == 0.0) {
      return points;
    }

    final _points = List<Vector<Float32x4>>(points.length);
    for (int i = 0; i < points.length; i++) {
      _points[i] = Float32x4VectorFactory.from(
        points[i].toList()
          ..insert(0, 1.0 * _interceptScale)
      );
    }
    return _points;
  }
}