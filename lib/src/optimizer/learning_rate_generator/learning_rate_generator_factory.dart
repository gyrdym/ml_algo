import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator.dart';

abstract class LearningRateGeneratorFactory {
  LearningRateGenerator decreasing();
  LearningRateGenerator constant();
  LearningRateGenerator fromType(LearningRateType type);
}
