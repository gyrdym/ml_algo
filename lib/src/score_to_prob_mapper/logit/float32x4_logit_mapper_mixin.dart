import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_linalg/matrix.dart';

class Float32x4LogitMapperMixin {
  final float32x4Zeroes = Float32x4.zero();
  final float32x4Ones = Float32x4.splat(1.0);

  MLMatrix float32x4ScoresToProbs(MLMatrix scores) =>
      scores.fastMap<Float32x4>(_scoreToProbFloat32x4);

  Float32x4 _scoreToProbFloat32x4(Float32x4 scores) =>
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