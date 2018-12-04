import 'package:ml_algo/src/optimizer/learning_rate_generator/decreasing.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/constant.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/generator.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/type.dart';

class LearningRateGeneratorFactory {
  static LearningRateGenerator decreasing() => DecreasingLearningRateGenerator();
  static LearningRateGenerator constant() => ConstantLearningRateGenerator();

  static LearningRateGenerator createByType(LearningRateType type) {
    LearningRateGenerator generator;
    switch (type) {
      case LearningRateType.constant:
        generator = constant();
        break;
      case LearningRateType.decreasing:
        generator = decreasing();
        break;
    }
    return generator;
  }
}