import 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';

class SimpleLearningRateGenerator implements LearningRateGenerator {
  double _initialValue;
  int _iterationCounter = 0;

  void init(double initialValue) {
    _initialValue = initialValue;
  }

  double getNextValue() => _initialValue / ++_iterationCounter;

  void stop() {
    _iterationCounter = 0;
  }
}
