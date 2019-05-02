import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
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
    Type dtype = DefaultParameterValues.dtype,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,

    // hidden arguments
    InterceptPreprocessorFactory interceptPreprocessorFactory =
        const InterceptPreprocessorFactoryImpl(),
    BatchSizeCalculator batchSizeCalculator = const BatchSizeCalculatorImpl(),
  })  : _optimizer = GradientOptimizer(
          interceptPreprocessorFactory.create(dtype, scale: fitIntercept
              ? interceptScale : 0.0)
              .addIntercept(trainingFeatures), trainingOutcomes,
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
  Matrix predict(Matrix features) => features * _weights;
}
