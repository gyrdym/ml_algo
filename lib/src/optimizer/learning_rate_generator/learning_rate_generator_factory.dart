import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/simple_learning_rate_generator.dart';

class LearningRateGeneratorFactory {
  static LearningRateGenerator Simple() => new SimpleLearningRateGenerator();
}