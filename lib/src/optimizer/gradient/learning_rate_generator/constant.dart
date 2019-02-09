import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';

class ConstantLearningRateGenerator implements LearningRateGenerator {
  double _initialValue;

  @override
  void init(double initialValue) {
    _initialValue = initialValue;
  }

  @override
  double getNextValue() => _initialValue;

  @override
  void stop() {}
}
