import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/link_function/logit/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:ml_linalg/dtype.dart';

class LinkFunctionFactoryImpl implements LinkFunctionFactory {
  const LinkFunctionFactoryImpl();

  @override
  LinkFunction createByType(LinkFunctionType type, {
    DType dtype = DType.float32,
  }) {
    switch (type) {
      case LinkFunctionType.inverseLogit:
        return InverseLogitLinkFunction(dtype);

      case LinkFunctionType.softmax:
        return SoftmaxLinkFunction(dtype);

      default:
        throw UnsupportedError('Unsupported link function type - $type');
    }
  }
}
