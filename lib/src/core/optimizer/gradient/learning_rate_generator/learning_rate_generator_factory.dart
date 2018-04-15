import 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/simple_learning_rate_generator.dart';

class LearningRateGeneratorFactory {
  static LearningRateGenerator createSimpleGenerator() => new SimpleLearningRateGenerator();
}