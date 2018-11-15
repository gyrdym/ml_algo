import 'dart:typed_data';

import 'package:dart_ml/src/cost_function/log_likelihood.dart';
import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/cost_function/squared.dart';

class CostFunctionFactory {
  static CostFunction<Float32x4List, Float32List, Float32x4> squared() => const SquaredCost();
  static CostFunction<Float32x4List, Float32List, Float32x4> logLikelihood() => const LogLikelihoodCost();
}
