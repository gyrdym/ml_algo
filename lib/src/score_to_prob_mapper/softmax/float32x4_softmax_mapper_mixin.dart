import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class Float32x4SoftmaxMapperMixin {
  MLMatrix float32x4ScoresToProbs(MLMatrix scores) {
    final maxValue = scores.max();
    final stableScores = scores - maxValue;
    final allProba = _toFloat32x4Probs(stableScores);
    final summedProba = allProba
        .reduceColumns(
            (MLVector resultColumn, MLVector scores) => resultColumn + scores);
    return allProba / summedProba;
  }

  MLMatrix _toFloat32x4Probs(MLMatrix scores) =>
      scores.fastMap<Float32x4>(_float32x4Logit);

  Float32x4 _float32x4Logit(Float32x4 scores) => Float32x4(
    //@TODO: find a more efficient way to raise exponent to the float power in SIMD way
    math.exp(scores.x),
    math.exp(scores.y),
    math.exp(scores.z),
    math.exp(scores.w),
  );
}