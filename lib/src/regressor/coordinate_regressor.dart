import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/regressor/_mixin/linear_regressor_mixin.dart';
import 'package:ml_algo/src/linear_optimizer/coordinate/coordinate.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';

class CoordinateRegressor with AssessablePredictorMixin, LinearRegressorMixin
    implements LinearRegressor {

  CoordinateRegressor(
      Matrix trainingFeatures,
      Matrix trainingOutcomes, {
        int iterationsLimit = ParameterDefaultValues.iterationsLimit,
        double minWeightsUpdate = ParameterDefaultValues.minCoefficientsUpdate,
        double lambda,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        DType dtype = ParameterDefaultValues.dtype,
        InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
        Matrix initialWeights,
        bool isTrainDataNormalized = false,
      }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        coefficients = CoordinateOptimizer(
          addInterceptIf(fitIntercept, trainingFeatures, interceptScale),
          trainingOutcomes,
          initialWeightsType: initialWeightsType,
          iterationsLimit: iterationsLimit,
          minCoefficientsUpdate: minWeightsUpdate,
          costFunction: const SquaredCost(),
          lambda: lambda,
          dtype: dtype,
          isFittingDataNormalized: isTrainDataNormalized,
        ).findExtrema(
          initialCoefficients: initialWeights,
          isMinimizingObjective: true,
        ).getRow(0);

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  @override
  final Vector coefficients;
}
