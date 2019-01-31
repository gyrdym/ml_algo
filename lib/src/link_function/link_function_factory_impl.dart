import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/link_function/logit_link_function/logit_link_function_factory.dart';

class ScoreToProbLinkFunctionFactoryImpl implements ScoreToProbLinkFunctionFactory {
  final LogitLinkFunctionFactory logitLinkFunctionFactory;

  const ScoreToProbLinkFunctionFactoryImpl({
    this.logitLinkFunctionFactory
  });

  @override
  LinkFunction fromType(LinkFunctionType type, Type dtype) {
    switch (type) {
      case LinkFunctionType.logit:
        return logitLinkFunctionFactory.fromDataType(dtype);
      default:
        throw UnimplementedError();
    }
  }
}
