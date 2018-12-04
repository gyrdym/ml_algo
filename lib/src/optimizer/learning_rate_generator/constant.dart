import 'package:ml_algo/src/optimizer/learning_rate_generator/generator.dart';

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