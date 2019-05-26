import 'package:ml_algo/src/optimizer/linear/gradient/learning_rate_generator/constant.dart';
import 'package:ml_algo/src/optimizer/linear/gradient/learning_rate_generator/decreasing.dart';
import 'package:ml_algo/src/optimizer/linear/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/optimizer/linear/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/linear/gradient/learning_rate_generator/learning_rate_type.dart';

class LearningRateGeneratorFactoryImpl implements LearningRateGeneratorFactory {
  const LearningRateGeneratorFactoryImpl();

  @override
  LearningRateGenerator decreasing() => DecreasingLearningRateGenerator();

  @override
  LearningRateGenerator constant() => ConstantLearningRateGenerator();

  @override
  LearningRateGenerator fromType(LearningRateType type) {
    switch (type) {
      case LearningRateType.constant:
        return constant();
      case LearningRateType.decreasing:
        return decreasing();
      default:
        throw UnimplementedError();
    }
  }
}
