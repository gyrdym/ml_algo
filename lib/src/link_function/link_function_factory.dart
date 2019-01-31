import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

abstract class ScoreToProbLinkFunctionFactory {
  LinkFunction fromType(LinkFunctionType type, Type dtype);
}