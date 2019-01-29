import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/metric_type.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/optimizer/gradient.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/gradient_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_linalg/linalg.dart';

class GradientRegressor implements LinearRegressor {
  final GradientOptimizer _optimizer;
  final InterceptPreprocessor _interceptPreprocessor;

  GradientRegressor({
    // public arguments
    int iterationLimit,
    double learningRate,
    double minWeightsUpdate,
    double lambda,
    GradientType gradientType = GradientType.stochastic,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    int randomSeed,
    int batchSize,
    Type dtype,
    LearningRateType learningRateType = LearningRateType.decreasing,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,

    // hidden arguments
    InterceptPreprocessorFactory interceptPreprocessorFactory = const InterceptPreprocessorFactoryImpl(),
  }) :
      _interceptPreprocessor = interceptPreprocessorFactory.create(dtype, scale: fitIntercept ? interceptScale : 0.0),
      _optimizer = GradientOptimizer(
        costFnType: CostFunctionType.squared,
        learningRateType: learningRateType,
        initialWeightsType: initialWeightsType,
        initialLearningRate: learningRate,
        minCoefficientsUpdate: minWeightsUpdate,
        iterationLimit: iterationLimit,
        lambda: lambda,
        batchSize: gradientType == GradientType.stochastic
            ? 1
            : gradientType == GradientType.miniBatch ? batchSize : double.maxFinite.toInt(),
        randomSeed: randomSeed,
      );

  @override
  MLVector get weights => _weights;
  MLVector _weights;

  @override
  void fit(MLMatrix features, MLVector labels, {MLVector initialWeights, bool isDataNormalized = false}) {
    _weights = _optimizer.findExtrema(_interceptPreprocessor.addIntercept(features), labels,
        initialWeights: initialWeights, isMinimizingObjective: true, arePointsNormalized: isDataNormalized);
  }

  @override
  double test(MLMatrix features, MLVector origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getError(prediction, origLabels);
  }

  MLVector predict(MLMatrix features) => (features * _weights).toVector();
}
