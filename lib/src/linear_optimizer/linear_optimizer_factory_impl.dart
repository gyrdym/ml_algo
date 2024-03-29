import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/closed_form_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/least_squares_coordinate_descent_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/gradient_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_iterable_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/newton_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/optimizer_to_regularization_mapping.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class LinearOptimizerFactoryImpl implements LinearOptimizerFactory {
  const LinearOptimizerFactoryImpl(
    this._initialCoefficientsGeneratorFactory,
    this._learningRateIterableFactory,
    this._convergenceDetectorFactory,
    this._randomizerFactory,
  );

  final InitialCoefficientsGeneratorFactory
      _initialCoefficientsGeneratorFactory;
  final LearningRateIterableFactory _learningRateIterableFactory;
  final ConvergenceDetectorFactory _convergenceDetectorFactory;
  final RandomizerFactory _randomizerFactory;

  @override
  LinearOptimizer createByType(
    LinearOptimizerType optimizerType,
    Matrix features,
    Matrix labels, {
    DType dtype = dTypeDefaultValue,
    required CostFunction costFunction,
    required LearningRateType learningRateType,
    required InitialCoefficientsType initialCoefficientsType,
    required double initialLearningRate,
    required double decay,
    required int dropRate,
    required double minCoefficientsUpdate,
    required int iterationLimit,
    required double lambda,
    required int batchSize,
    RegularizationType? regularizationType,
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
          features,
          labels,
          costFunction: costFunction,
          lambda: lambda,
          batchSize: batchSize,
          dtype: dtype,
          learningRates: _learningRateIterableFactory.fromType(
            type: learningRateType,
            initialValue: initialLearningRate,
            decay: decay,
            dropRate: dropRate,
            iterationLimit: iterationLimit,
          ),
          initialCoefficientsGenerator: _initialCoefficientsGeneratorFactory
              .fromType(initialCoefficientsType, dtype),
          minCoefficientsUpdate: minCoefficientsUpdate,
          randomizer: _randomizerFactory.create(randomSeed),
        );

      case LinearOptimizerType.coordinate:
        return LeastSquaresCoordinateDescentOptimizer(
          features,
          labels,
          dtype: dtype,
          lambda: lambda,
          initialCoefficientsGenerator: _initialCoefficientsGeneratorFactory
              .fromType(initialCoefficientsType, dtype),
          convergenceDetector: _convergenceDetectorFactory.create(
              minCoefficientsUpdate, iterationLimit),
          isFittingDataNormalized: isFittingDataNormalized,
        );

      case LinearOptimizerType.newton:
        return NewtonOptimizer(
          features: features,
          labels: labels,
          costFunction: costFunction,
          iterationLimit: iterationLimit,
          lambda: lambda,
          minCoefficientsUpdate: minCoefficientsUpdate,
          dtype: dtype,
        );

      case LinearOptimizerType.closedForm:
        return ClosedFormOptimizer(features, labels);

      default:
        throw UnsupportedError(
            'Unsupported linear optimizer type - $optimizerType');
    }
  }
}
