import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:quiver/collection.dart';

final learningRateTypeToEncodedValue = BiMap<LearningRateType, String>()
  ..addAll({
    LearningRateType.constant: 'C',
    LearningRateType.decreasingAdaptive: 'DA',
    LearningRateType.timeBased: 'TB',
    LearningRateType.exponential: 'E',
  });
