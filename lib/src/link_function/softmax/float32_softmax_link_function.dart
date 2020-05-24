import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/softmax_link_function_mixin.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class Float32SoftmaxLinkFunction
    with
        SoftmaxLinkFunctionMixin
    implements
        SoftmaxLinkFunction {

  const Float32SoftmaxLinkFunction();

  @override
  final DType dtype = DType.float32;

  @override
  Matrix getNumerator(Matrix scores) =>
      scores.fastMap<Float32x4>(_raiseExponentToScores);

  Float32x4 _raiseExponentToScores(Float32x4 scores) => Float32x4(
    //@TODO: find a more efficient way to raise exponent to the float power in SIMD way
    math.exp(scores.x),
    math.exp(scores.y),
    math.exp(scores.z),
    math.exp(scores.w),
  );
}
