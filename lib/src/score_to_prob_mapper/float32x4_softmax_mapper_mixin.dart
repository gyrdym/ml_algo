import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class Float32x4SoftmaxMapperMixin {
  MLVector float32x4ScoresToProbs(MLVector scores, MLMatrix scoresByClasses) {
    final maxValue = scoresByClasses.max();
    return (_toFloat32x4Probs(scores - maxValue)) / (scoresByClasses - maxValue)
        .reduceColumns((MLVector resultColumn, MLVector scores) {
      resultColumn += _toFloat32x4Probs(scores);
      return resultColumn;
    }, initValue: MLVector.zero(scores.length));
  }

  MLVector _toFloat32x4Probs(MLVector scores) =>
      scores.fastMap<Float32x4>(
              (Float32x4 score, int startOffset, int endOffset) =>
              _float32x4Logit(score));

  Float32x4 _float32x4Logit(Float32x4 scores) => Float32x4(
    //@TODO: find a more efficient way to raise exponent to the float power in SIMD way
    math.exp(scores.x),
    math.exp(scores.y),
    math.exp(scores.z),
    math.exp(scores.w),
  );
}