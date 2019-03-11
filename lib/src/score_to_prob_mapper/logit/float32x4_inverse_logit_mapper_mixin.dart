import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_linalg/matrix.dart';

mixin Float32x4InverseLogitMapper {
  static Float32x4 _ones = Float32x4.splat(1.0);
  static Float32x4 _upperBound = Float32x4.splat(10);
  static Float32x4 _lowerBound = Float32x4.splat(-10);

  Matrix getFloat32x4Probabilities(Matrix scores) {
    // binary classification case
    if (scores.columnsNum == 1) {
      final scoresVector = scores.getColumn(0);
      return Matrix
          .columns([scoresVector.fastMap<Float32x4>(scoreToProb)]);
    }

    // multi class classification case
    return scores.fastMap<Float32x4>((score) => scoreToProb(score, null, null));
  }

  Float32x4 scoreToProb(Float32x4 scores, int start, int end) {
    final exp = _exp(scores);
    final bigMask = scores.greaterThanOrEqual(_upperBound);
    final smallMask = scores.lessThanOrEqual(_lowerBound);
    final unsafeProbs = exp / (_ones + exp);
    final big = bigMask.select(_ones, unsafeProbs);
    return smallMask.select(Float32x4.zero(), big);
  }

  Float32x4 _exp(Float32x4 value) =>
    Float32x4(
      // TODO find a more efficient way to raise exponent to the float power in SIMD way
      math.exp(value.x),
      math.exp(value.y),
      math.exp(value.z),
      math.exp(value.w),
    );
}