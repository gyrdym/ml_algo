import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/link_function/link_function.dart';

class LogitLinkFunction implements LinkFunction {
  final float32x4Zeroes = Float32x4.zero();
  final float32x4Ones = Float32x4.splat(1.0);

  @override
  Float32x4 float32x4Link(Float32x4 scores) =>
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
