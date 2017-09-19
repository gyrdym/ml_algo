part of 'package:dart_ml/src/implementation.dart';

class _SimpleLearningRateGeneratorImpl implements LearningRateGenerator {
  double _coef;

  void init(double initialValue) {
    _coef = initialValue;
  }

  double generate(int iterationNumber) {
    return _coef / iterationNumber;
  }
}