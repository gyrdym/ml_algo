import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_type.dart';

abstract class LearningRateGeneratorFactory {
  LearningRateGenerator decreasing();
  LearningRateGenerator constant();
  LearningRateGenerator fromType(LearningRateType type);
}
