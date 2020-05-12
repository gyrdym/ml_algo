import 'package:ml_algo/src/link_function/link_function_factory_impl.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/link_function/logit/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:test/test.dart';

void main() {
  group('LinkFunctionFactoryImpl', () {
    test('should create inverse logit link function instance', () {
      final factory = const LinkFunctionFactoryImpl();
      final linkFunction = factory.createByType(LinkFunctionType.inverseLogit);
      expect(linkFunction, isA<InverseLogitLinkFunction>());
    });

    test('should create softmax link function instance', () {
      final factory = const LinkFunctionFactoryImpl();
      final linkFunction = factory.createByType(LinkFunctionType.softmax);
      expect(linkFunction, isA<SoftmaxLinkFunction>());
    });

    test('should throw an error if null is passed as a type', () {
      final factory = const LinkFunctionFactoryImpl();
      final actual = () => factory.createByType(null);
      expect(actual, throwsUnsupportedError);
    });
  });
}
