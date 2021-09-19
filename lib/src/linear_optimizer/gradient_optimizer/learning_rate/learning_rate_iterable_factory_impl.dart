import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/constant.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/exponential.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/step_based.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/time_based.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_iterable_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';

class LearningRateIterableFactoryImpl implements LearningRateIterableFactory {
  const LearningRateIterableFactoryImpl();

  @override
  Iterable<double> fromType({
    required LearningRateType type,
    required double initialValue,
    required double decay,
    required int dropRate,
    required int iterationLimit,
  }) {
    switch (type) {
      case LearningRateType.constant:
        return ConstantLearningRateIterable(initialValue, iterationLimit);

      case LearningRateType.decreasingAdaptive:
      case LearningRateType.timeBased:
        return TimeBasedLearningRateIterable(
          initialValue: initialValue,
          decay: decay,
          limit: iterationLimit,
        );

      case LearningRateType.exponential:
        return ExponentialLearningRateIterable(
          initialValue: initialValue,
          decay: decay,
          limit: iterationLimit,
        );

      case LearningRateType.stepBased:
        return StepBasedLearningRateIterable(
          initialValue: initialValue,
          decay: decay,
          dropRate: dropRate,
          limit: iterationLimit,
        );

      default:
        throw UnsupportedError('Unsupported learning rate type - $type');
    }
  }
}
