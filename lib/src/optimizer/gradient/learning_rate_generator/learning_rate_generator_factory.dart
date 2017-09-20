part of 'package:dart_ml/src/implementation.dart';

class LearningRateGeneratorFactory {
  static LearningRateGenerator createSimpleGenerator() => new _SimpleLearningRateGeneratorImpl();
}