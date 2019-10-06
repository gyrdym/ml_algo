import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_linalg/dtype.dart';

abstract class LinkFunctionFactory {
  LinkFunction createByType(LinkFunctionType type, {
    DType dtype = DType.float32,
  });
}
