import 'package:dart_ml/src/cost_function/log_likelihood.dart';
import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/cost_function/squared.dart';

class CostFunctionFactory {
  static CostFunction Squared() => const SquaredCost();
  static CostFunction LogLikelihood() => const LogLikelihoodCost();
}
