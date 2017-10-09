part of 'package:dart_ml/src/core/implementation.dart';

class LearningRateGeneratorFactory {
  static LearningRateGenerator createSimpleGenerator() => new _SimpleLearningRateGenerator();
}