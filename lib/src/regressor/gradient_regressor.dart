import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/helpers/add_intercept.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/gradient/gradient.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/gradient_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class GradientRegressor implements LinearRegressor {
  GradientRegressor(this.trainingFeatures, this.trainingOutcomes, {
    // public arguments
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double initialLearningRate = DefaultParameterValues.initialLearningRate,
    double minWeightsUpdate = DefaultParameterValues.minCoefficientsUpdate,
    double lambda,
    GradientType gradientType = GradientType.stochastic,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    int randomSeed,
    int batchSize = 1,
    DType dtype = DefaultParameterValues.dtype,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,

    // hidden arguments
    BatchSizeCalculator batchSizeCalculator = const BatchSizeCalculatorImpl(),
  })  : fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        _optimizer = GradientOptimizer(
          addInterceptIf(trainingFeatures, fitIntercept, interceptScale),
          trainingOutcomes,
          costFnType: CostFunctionType.squared,
          learningRateType: learningRateType,
          initialWeightsType: initialWeightsType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minWeightsUpdate,
          iterationLimit: iterationsLimit,
          lambda: lambda,
          batchSize: batchSizeCalculator.calculate(gradientType, batchSize),
          randomSeed: randomSeed,
        );

  @override
  final Matrix trainingFeatures;

  @override
  final Matrix trainingOutcomes;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  final GradientOptimizer _optimizer;

  @override
  Vector get weights => _weights;
  Vector _weights;

  @override
  void fit({Matrix initialWeights}) {
    _weights = _optimizer.findExtrema(
      initialWeights: initialWeights?.transpose(),
      isMinimizingObjective: true,
    ).getColumn(0);
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getScore(prediction, origLabels);
  }

  @override
  Matrix predict(Matrix features) =>
      addInterceptIf(features, fitIntercept, interceptScale) * _weights;
}
