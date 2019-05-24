import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_link_function_mixin.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class InverseLogitLinkFunction with Float32InverseLogitLinkFunction
    implements LinkFunction {

  InverseLogitLinkFunction(this.dtype);

  final DType dtype;

  @override
  Matrix link(Matrix scores) {
    switch (dtype) {
      case DType.float32:
        return getFloat32x4Probabilities(scores);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}
