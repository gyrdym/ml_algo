import 'package:dart_ml/src/optimizer/learning_rate_generator/decreasing.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/constant.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/generator.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/type.dart';

class LearningRateGeneratorFactory {
  static LearningRateGenerator Decreasing() => new DecreasingLearningRateGenerator();
  static LearningRateGenerator Constant() => new ConstantLearningRateGenerator();

  static LearningRateGenerator createByType(LearningRateType type) {
    LearningRateGenerator generator;
    switch (type) {
      case LearningRateType.constant:
        generator = Constant();
        break;
      case LearningRateType.decreasing:
        generator = Decreasing();
        break;
    }
    return generator;
  }
}