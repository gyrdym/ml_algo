import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/regressor/_mixin/linear_regressor_mixin.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_algo/src/solver/linear/gradient/gradient.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class GradientRegressor with AssessablePredictorMixin, LinearRegressorMixin
    implements LinearRegressor {

  GradientRegressor(
      Matrix trainingFeatures,
      Matrix trainingOutcomes, {
        int iterationsLimit = ParameterDefaultValues.iterationsLimit,
        double initialLearningRate = ParameterDefaultValues.initialLearningRate,
        double minWeightsUpdate = ParameterDefaultValues.minCoefficientsUpdate,
        double lambda,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        int randomSeed,
        int batchSize = 1,
        DType dtype = ParameterDefaultValues.dtype,
        Matrix initialWeights,
        LearningRateType learningRateType = LearningRateType.constant,
        InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
      }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        coefficients = GradientOptimizer(
          addInterceptIf(fitIntercept, trainingFeatures, interceptScale),
          trainingOutcomes,
          costFunction: const SquaredCost(),
          learningRateType: learningRateType,
          initialWeightsType: initialWeightsType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minWeightsUpdate,
          iterationLimit: iterationsLimit,
          lambda: lambda,
          batchSize: batchSize,
          randomSeed: randomSeed,
        ).findExtrema(
          initialWeights: initialWeights?.transpose(),
          isMinimizingObjective: true,
        ).getColumn(0);

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  @override
  final Vector coefficients;
}
