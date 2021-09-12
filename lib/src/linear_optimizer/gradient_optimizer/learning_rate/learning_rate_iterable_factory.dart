import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';

abstract class LearningRateIterableFactory {
  Iterable<double> fromType({
    required LearningRateType type,
    required double initialValue,
    required double decay,
    required int iterationLimit,
  });
}
