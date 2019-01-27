import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';

final zeroes = Float32x4.splat(0.0);
final ones = Float32x4.splat(1.0);

ScoreToProbLinkFunction<Float32x4> vectorizedLogitLink = (Float32x4 scores) =>
    ones /
    (ones +
        Float32x4(
          //@TODO: find a more efficient way to raise exponent to the float power in SIMD way
          math.exp(-scores.x),
          math.exp(-scores.y),
          math.exp(-scores.z),
          math.exp(-scores.w),
        ));

IndicatorFunction<Float32x4> vectorizedIndicator =
    (Float32x4 labels, Float32x4 target) => labels.equal(target).select(ones, zeroes);
