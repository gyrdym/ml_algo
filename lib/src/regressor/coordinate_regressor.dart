import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/solver/linear/coordinate/coordinate.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';

class CoordinateRegressor implements LinearRegressor {
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
        _fitIntercept = fitIntercept,
        _interceptScale = interceptScale,
        coefficients = CoordinateOptimizer(
          addInterceptIf(fitIntercept, trainingFeatures, interceptScale),
          trainingOutcomes,
          initialWeightsType: initialWeightsType,
          iterationsLimit: iterationsLimit,
          minCoefficientsDiff: minWeightsUpdate,
          costFunction: const SquaredCost(),
          lambda: lambda,
          dtype: dtype,
          isTrainDataNormalized: isTrainDataNormalized,
        ).findExtrema(
          initialWeights: initialWeights,
          isMinimizingObjective: true,
        ).getRow(0);

  final bool _fitIntercept;

  final double _interceptScale;

  @override
  final Vector coefficients;

  @override
  double assess(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getScore(prediction, origLabels);
  }

  @override
  Matrix predict(Matrix features) =>
      addInterceptIf(_fitIntercept, features, _interceptScale) * coefficients;
}
