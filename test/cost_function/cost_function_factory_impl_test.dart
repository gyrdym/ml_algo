import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:test/test.dart';

void main() {
  group('CostFunctionFactoryImpl', () {
    final factory = const CostFunctionFactoryImpl();

    test('should create a squared cost function', () {
      final costFn = factory.createByType(CostFunctionType.squared);
      expect(costFn, isA<SquaredCost>());
    });

    test('should create a loglikelihood cost function considering passed '
        'link function', () {
      final linkFn = const Float32InverseLogitLinkFunction();

      final costFn = factory.createByType(
        CostFunctionType.logLikelihood,
        linkFunction: linkFn,
      );

      expect(costFn, isA<LogLikelihoodCost>());
    });

    test('should throw an exception if no link function provided for '
        'loglikelihood function', () {
      expect(
        () => factory.createByType(CostFunctionType.logLikelihood),
        throwsException,
      );
    });
  });
}
