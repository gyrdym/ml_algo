import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/softmax_link_function_mixin.dart';
import 'package:ml_linalg/matrix.dart';

class Float64SoftmaxLinkFunction
    with
        SoftmaxLinkFunctionMixin
    implements
        SoftmaxLinkFunction {

  const Float64SoftmaxLinkFunction();

  @override
  Matrix getNumerator(Matrix scores) =>
      scores.fastMap<Float64x2>(_raiseExponentToScores);

  Float64x2 _raiseExponentToScores(Float64x2 scores) => Float64x2(
    //@TODO: find a more efficient way to raise exponent to the float power in SIMD way
    math.exp(scores.x),
    math.exp(scores.y),
  );
}
