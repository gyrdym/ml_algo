import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/constant.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/decreasing.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';

class LearningRateGeneratorFactoryImpl implements LearningRateGeneratorFactory {
  const LearningRateGeneratorFactoryImpl();

  @override
  LearningRateGenerator fromType(LearningRateType type) {
    switch (type) {
      case LearningRateType.constant:
        return ConstantLearningRateGenerator();

      case LearningRateType.decreasingAdaptive:
        return DecreasingLearningRateGenerator();

      default:
        throw UnsupportedError('Unsupported learning rate type - $type');
    }
  }
}
