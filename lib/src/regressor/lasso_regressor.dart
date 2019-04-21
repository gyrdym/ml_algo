import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/coordinate/coordinate.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_linalg/linalg.dart';

class LassoRegressor implements LinearRegressor {
  LassoRegressor(this.trainingFeatures, this.trainingOutcomes, {
    // public arguments
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double minWeightsUpdate = DefaultParameterValues.minCoefficientsUpdate,
    double lambda,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    Type dtype = DefaultParameterValues.dtype,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    bool isTrainDataNormalized = false,

    // hidden arguments
    InterceptPreprocessorFactory interceptPreprocessorFactory =
        const InterceptPreprocessorFactoryImpl(),
  }) : _interceptPreprocessor = interceptPreprocessorFactory.create(dtype,
            scale: fitIntercept ? interceptScale : 0.0),
        _optimizer = CoordinateOptimizer(
          initialWeightsType: initialWeightsType,
          costFunctionType: CostFunctionType.squared,
          iterationsLimit: iterationsLimit,
          minCoefficientsDiff: minWeightsUpdate,
          lambda: lambda,
          dtype: dtype,
          isTrainDataNormalized: isTrainDataNormalized,
        );

  @override
  final Matrix trainingFeatures;

  @override
  final Matrix trainingOutcomes;

  final CoordinateOptimizer _optimizer;
  final InterceptPreprocessor _interceptPreprocessor;

  @override
  Vector get weights => _weights;
  Vector _weights;

  @override
  void fit({Matrix initialWeights}) {
    _weights = _optimizer.findExtrema(
      _interceptPreprocessor.addIntercept(trainingFeatures),
      trainingOutcomes,
      initialWeights: initialWeights,
      isMinimizingObjective: true,
    ).getRow(0);
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
