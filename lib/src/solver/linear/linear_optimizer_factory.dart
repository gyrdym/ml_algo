import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer_type.dart';
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
            InitialWeightsType initialWeightsType,
            double initialLearningRate,
            double minCoefficientsUpdate,
            int iterationLimit,
            double lambda,
            int batchSize,
            int randomSeed,
            bool isFittingDataNormalized,
      });
}
