import 'dart:typed_data';
import 'dart:math' as math;

import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/logit/logit_scores_matrix_dimension_exception.dart';
import 'package:ml_linalg/matrix.dart';

class Float32InverseLogitLinkFunction implements LinkFunction {
  const Float32InverseLogitLinkFunction();

  static final Float32x4 _simdOnes = Float32x4.splat(1.0);
  static final Float32x4 _simdZeroes = Float32x4.zero();
  static final Float32x4 _upperBound = Float32x4.splat(10);
  static final Float32x4 _lowerBound = Float32x4.splat(-10);

  @override
  Matrix link(Matrix scores) {
    if (scores.columnsNum != 1) {
      throw LogitScoresMatrixDimensionException(scores.columnsNum);
    }

    final scoresVector = scores.getColumn(0);

    return Matrix
        .fromColumns([scoresVector.fastMap<Float32x4>(scoresToProbabilities)]);
  }

  Float32x4 scoresToProbabilities(Float32x4 scores) {
    final exponentToScores = _raiseExponentToScores(scores);

    final upperBoundedMask = scores.greaterThanOrEqual(_upperBound);
    final lowerBoundedMask = scores.lessThanOrEqual(_lowerBound);

    final unsafeProbabilities = exponentToScores /
        (_simdOnes + exponentToScores);

    final safeUpperBoundedProbabilities = upperBoundedMask
        .select(_simdOnes, unsafeProbabilities);
    final safeLowerBoundedProbabilities = lowerBoundedMask
        .select(_simdZeroes, safeUpperBoundedProbabilities);

    return safeLowerBoundedProbabilities;
  }

  Float32x4 _raiseExponentToScores(Float32x4 scores) =>
      Float32x4(
        // TODO find a more efficient way to raise exponent to the float power in SIMD way
        math.exp(scores.x),
        math.exp(scores.y),
        math.exp(scores.z),
        math.exp(scores.w),
      );
}
