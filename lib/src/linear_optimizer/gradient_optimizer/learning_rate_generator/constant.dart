import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator.dart';

class ConstantLearningRateGenerator implements LearningRateGenerator {
  late double _initialValue;

  @override
  void init(double initialValue) {
    _initialValue = initialValue;
  }

  @override
  double getNextValue() => _initialValue;

  @override
  void stop() {}
}
