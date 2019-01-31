import 'package:ml_algo/src/link_function/link_function.dart';

abstract class LogitLinkFunctionFactory {
  LinkFunction fromDataType(Type dtype);
}
