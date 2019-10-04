import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';

const optimizerToRegularization = {
  LinearOptimizerType.vanillaCD: [RegularizationType.L1],
  LinearOptimizerType.vanillaGD: [RegularizationType.L2],
};