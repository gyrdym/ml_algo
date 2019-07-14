import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/solver/linear/gradient/gradient.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class GradientRegressor implements LinearRegressor {
  GradientRegressor(
      Matrix trainingFeatures,
      Matrix trainingOutcomes, {
        int iterationsLimit = DefaultParameterValues.iterationsLimit,
        double initialLearningRate = DefaultParameterValues.initialLearningRate,
        double minWeightsUpdate = DefaultParameterValues.minCoefficientsUpdate,
        double lambda,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        int randomSeed,
        int batchSize = 1,
        DType dtype = DefaultParameterValues.dtype,
        Matrix initialWeights,
        LearningRateType learningRateType = LearningRateType.constant,
        InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
      }) :
        _fitIntercept = fitIntercept,
        _interceptScale = interceptScale,
        coefficients = GradientOptimizer(
          addInterceptIf(fitIntercept, trainingFeatures, interceptScale),
          trainingOutcomes,
          costFunction: SquaredCost(),
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
