import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function_mixin.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class SoftmaxLinkFunction with Float32SoftmaxLinkFunction
    implements LinkFunction {

  SoftmaxLinkFunction(this.dtype);

  final DType dtype;

  @override
  Matrix link(Matrix scores) {
    switch (dtype) {
      case DType.float32:
        return float32x4ScoresToProbs(scores);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}