import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/helpers/add_intercept.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/gradient/gradient.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class GradientRegressor implements LinearRegressor {
  GradientRegressor(
      this.trainingFeatures,
      this.trainingOutcomes, {
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
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        weights = GradientOptimizer(
          addInterceptIf(fitIntercept, trainingFeatures, interceptScale),
          trainingOutcomes,
          costFnType: CostFunctionType.squared,
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
  final Matrix trainingFeatures;

  @override
  final Matrix trainingOutcomes;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  @override
  final Vector weights;

  @override
  double test(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getScore(prediction, origLabels);
  }

  @override
  Matrix predict(Matrix features) =>
      addInterceptIf(fitIntercept, features, interceptScale) * weights;
}
