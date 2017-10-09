part of 'package:dart_ml/src/core/implementation.dart';

class _SimpleLearningRateGenerator implements LearningRateGenerator {
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
