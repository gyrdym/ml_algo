import 'dart:typed_data';

import 'package:dart_ml/src/data_preprocessing/intercept_preprocessor.dart';
import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:linalg/vector.dart';

abstract class LinearRegressor implements Evaluable<Float32x4, Float32x4List, Float32List> {

  final Optimizer<Float32x4, Float32x4List, Float32List> _optimizer;
  final InterceptPreprocessor _interceptPreprocessor;

  LinearRegressor(this._optimizer, double interceptScale) :
    _interceptPreprocessor = InterceptPreprocessor(interceptScale: interceptScale);

  SIMDVector<Float32x4List, Float32List, Float32x4> get weights => _weights;
  SIMDVector<Float32x4List, Float32List, Float32x4> _weights;

  @override
  void fit(
    List<SIMDVector<Float32x4List, Float32List, Float32x4>> features,
    SIMDVector<Float32x4List, Float32List, Float32x4> labels,
    {
      SIMDVector<Float32x4List, Float32List, Float32x4> initialWeights,
      bool isDataNormalized = false
    }
  ) {
    _weights = _optimizer.findExtrema(
      _interceptPreprocessor.addIntercept(features),
      labels,
      initialWeights: initialWeights,
      isMinimizingObjective: true,
      arePointsNormalized: isDataNormalized
    );
  }

  @override
  double test(
    List<SIMDVector<Float32x4List, Float32List, Float32x4>> features,
    SIMDVector<Float32x4List, Float32List, Float32x4> origLabels,
    MetricType metricType
  ) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getError(prediction, origLabels);
  }

  SIMDVector<Float32x4List, Float32List, Float32x4> predict(
    List<SIMDVector<Float32x4List, Float32List, Float32x4>> features
  ) {
    final labels = List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }
    return Float32x4VectorFactory.from(labels);
  }
}
