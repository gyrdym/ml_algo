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
        DType dtype,
        CostFunction costFunction,
        LearningRateType learningRateType,
        InitialCoefficientsType initialWeightsType,
        double initialLearningRate,
        double minCoefficientsUpdate,
        int iterationLimit,
        double lambda,
        RegularizationType regularizationType,
        int batchSize,
        int randomSeed,
        bool isFittingDataNormalized,
      });
}
