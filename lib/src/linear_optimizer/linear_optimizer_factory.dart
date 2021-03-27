import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LinearOptimizerFactory {
  LinearOptimizer createByType(
      LinearOptimizerType optimizerType,
      Matrix points,
      Matrix labels, {
            required DType dtype,
            required CostFunction costFunction,
            required LearningRateType learningRateType,
            required InitialCoefficientsType initialCoefficientsType,
            required double initialLearningRate,
            required double minCoefficientsUpdate,
            required int iterationLimit,
            required double lambda,
            RegularizationType? regularizationType,
            required int batchSize,
            int? randomSeed,
            required bool isFittingDataNormalized,
      });
}
