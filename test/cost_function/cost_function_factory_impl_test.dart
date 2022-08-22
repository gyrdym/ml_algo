import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/least_square_cost_function.dart';
import 'package:ml_algo/src/cost_function/log_likelihood_cost_function.dart';
import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('CostFunctionFactoryImpl', () {
    final factory = const CostFunctionFactoryImpl();
    final linkFn = const InverseLogitLinkFunction();
    final positiveLabel = 100;
    final negativeLabel = 0;

    test('should create a squared cost function', () {
      final costFn = factory.createByType(CostFunctionType.leastSquare,
          dtype: DType.float32);
      expect(costFn, isA<LeastSquareCostFunction>());
    });

    test(
        'should create a loglikelihood cost function considering passed '
        'link function, positive and negative labels', () {
      final costFn = factory.createByType(
        CostFunctionType.logLikelihood,
        linkFunction: linkFn,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        dtype: DType.float32,
      );

      expect(costFn, isA<LogLikelihoodCostFunction>());
    });

    test(
        'should throw an exception if no link function provided for '
        'loglikelihood function', () {
      expect(
        () => factory.createByType(
          CostFunctionType.logLikelihood,
          positiveLabel: positiveLabel,
          negativeLabel: negativeLabel,
        ),
        throwsException,
      );
    });

    test(
        'should throw an exception if no positive label provided for '
        'loglikelihood function', () {
      expect(
        () => factory.createByType(
          CostFunctionType.logLikelihood,
          linkFunction: linkFn,
          negativeLabel: negativeLabel,
        ),
        throwsException,
      );
    });

    test(
        'should throw an exception if no negative label provided for '
        'loglikelihood function', () {
      expect(
        () => factory.createByType(
          CostFunctionType.logLikelihood,
          linkFunction: linkFn,
          positiveLabel: positiveLabel,
        ),
        throwsException,
      );
    });
  });
}
