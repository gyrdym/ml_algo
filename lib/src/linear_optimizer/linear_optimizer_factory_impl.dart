import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/coordinate_optimizer/coordinate_descent_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/gradient_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/optimizer_to_regularization_mapping.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class LinearOptimizerFactoryImpl implements LinearOptimizerFactory {
  const LinearOptimizerFactoryImpl();

  @override
  LinearOptimizer createByType(
      LinearOptimizerType optimizerType,
      Matrix fittingPoints,
      Matrix fittingLabels, {
        DType dtype = DType.float32,
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
      }) {
    if (regularizationType != null &&
        !optimizerToRegularization[optimizerType]
            .contains(regularizationType)) {
      throw UnsupportedError('Regularization type $regularizationType is '
          'unsupported by optimizer $optimizerType');
    }

    switch (optimizerType) {

      case LinearOptimizerType.vanillaGD:
        return GradientOptimizer(
          fittingPoints, fittingLabels,
          costFunction: costFunction,
          learningRateType: learningRateType,
          initialCoefficientsType: initialWeightsType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minCoefficientsUpdate,
          iterationLimit: iterationLimit,
          lambda: lambda,
          batchSize: batchSize,
          randomSeed: randomSeed,
          dtype: dtype,
        );

      case LinearOptimizerType.vanillaCD:
        return CoordinateDescentOptimizer(
          fittingPoints, fittingLabels,
          dtype: dtype,
          costFunction: costFunction,
          minCoefficientsUpdate: minCoefficientsUpdate,
          iterationsLimit: iterationLimit,
          lambda: lambda,
          initialWeightsType: initialWeightsType,
          isFittingDataNormalized: isFittingDataNormalized,
        );

      default:
        throw UnsupportedError(
            'Unsupported linear optimizer type - $optimizerType');
    }
  }
}
