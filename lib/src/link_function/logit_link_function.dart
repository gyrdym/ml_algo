import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogitLinkFunction implements LinkFunction {
  final Type dtype;

  final float32x4Zeroes = Float32x4.zero();
  final float32x4Ones = Float32x4.splat(1.0);

  LogitLinkFunction(this.dtype);

  @override
  MLVector linkScoresToProbs(MLVector scores, [MLMatrix scoresByClasses]) {
    switch (dtype) {
      case Float32x4:
        return scores.fastMap<Float32x4>(
            (Float32x4 el, int startOffset, int endOffset) =>
                scoreToProbFloat32x4(el));
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }

  Float32x4 scoreToProbFloat32x4(Float32x4 scores) =>
      float32x4Ones /
      (float32x4Ones +
          Float32x4(
            //@TODO: find a more efficient way to raise exponent to the float power in SIMD way
            math.exp(-scores.x),
            math.exp(-scores.y),
            math.exp(-scores.z),
            math.exp(-scores.w),
          ));
}
