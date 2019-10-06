import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';

const optimizerToRegularization = {
  LinearOptimizerType.coordinate: [RegularizationType.L1],
  LinearOptimizerType.gradient: [RegularizationType.L2],
};