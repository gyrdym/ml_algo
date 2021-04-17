import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/coordinate_optimizer/coordinate_descent_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/gradient_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/optimizer_to_regularization_mapping.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class LinearOptimizerFactoryImpl implements LinearOptimizerFactory {
  const LinearOptimizerFactoryImpl(
      this._initialCoefficientsGeneratorFactory,
      this._learningRateGeneratorFactory,
      this._convergenceDetectorFactory,
      this._randomizerFactory,
  );

  final InitialCoefficientsGeneratorFactory _initialCoefficientsGeneratorFactory;
  final LearningRateGeneratorFactory _learningRateGeneratorFactory;
  final ConvergenceDetectorFactory _convergenceDetectorFactory;
  final RandomizerFactory _randomizerFactory;

  @override
  LinearOptimizer createByType(
      LinearOptimizerType optimizerType,
      Matrix fittingPoints,
      Matrix fittingLabels, {
        DType dtype = DType.float32,
        required CostFunction costFunction,
        LearningRateType learningRateType = LearningRateType.decreasingAdaptive,
        InitialCoefficientsType initialCoefficientsType = InitialCoefficientsType.zeroes,
        double initialLearningRate = 1e-3,
        double minCoefficientsUpdate = 1e-12,
        int iterationLimit = 100,
        required double lambda,
        RegularizationType? regularizationType,
        required int batchSize,
        int? randomSeed,
        required bool isFittingDataNormalized,
      }) {

    if (regularizationType != null &&
        !optimizerToRegularization[optimizerType]!
            .contains(regularizationType)) {
      throw UnsupportedError('Regularization type $regularizationType is '
          'unsupported by optimizer $optimizerType');
    }

    switch (optimizerType) {

      case LinearOptimizerType.gradient:
        return GradientOptimizer(
          fittingPoints,
          fittingLabels,
          costFunction: costFunction,
          initialLearningRate: initialLearningRate,
          lambda: lambda,
          batchSize: batchSize,
          dtype: dtype,
          learningRateGenerator: _learningRateGeneratorFactory
              .fromType(learningRateType),
          initialCoefficientsGenerator: _initialCoefficientsGeneratorFactory
              .fromType(initialCoefficientsType, dtype),
          convergenceDetector: _convergenceDetectorFactory
              .create(minCoefficientsUpdate, iterationLimit),
          randomizer: _randomizerFactory.create(randomSeed),
        );

      case LinearOptimizerType.coordinate:
        return CoordinateDescentOptimizer(
          fittingPoints,
          fittingLabels,
          dtype: dtype,
          costFunction: costFunction,
          lambda: lambda,
          initialCoefficientsGenerator: _initialCoefficientsGeneratorFactory
              .fromType(initialCoefficientsType, dtype),
          convergenceDetector: _convergenceDetectorFactory
              .create(minCoefficientsUpdate, iterationLimit),
          isFittingDataNormalized: isFittingDataNormalized,
        );

      default:
        throw UnsupportedError(
            'Unsupported linear optimizer type - $optimizerType');
    }
  }
}
