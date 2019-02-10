import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

abstract class LinkFunctionFactory {
  LinkFunction fromType(LinkFunctionType type);
}
