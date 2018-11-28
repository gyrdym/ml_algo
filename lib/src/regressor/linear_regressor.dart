import 'dart:typed_data';

import 'package:dart_ml/src/data_preprocessing/intercept_preprocessor.dart';
import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:linalg/linalg.dart';

abstract class LinearRegressor implements Evaluable<Float32x4, Vector<Float32x4>> {
  final Optimizer<Float32x4, Vector<Float32x4>> _optimizer;
  final InterceptPreprocessor _interceptPreprocessor;

  LinearRegressor(this._optimizer, double interceptScale) :
    _interceptPreprocessor = InterceptPreprocessor(interceptScale: interceptScale);

  Vector<Float32x4> get weights => _weights;
  Vector<Float32x4> _weights;

  @override
  void fit(Matrix<Float32x4, Vector<Float32x4>> features, Vector<Float32x4> labels, {
      Vector<Float32x4> initialWeights,
      bool isDataNormalized = false
    }) {
    _weights = _optimizer.findExtrema(
      _interceptPreprocessor.addIntercept(features),
      labels,
      initialWeights: initialWeights,
      isMinimizingObjective: true,
      arePointsNormalized: isDataNormalized
    );
  }

  @override
  double test(Matrix<Float32x4, Vector<Float32x4>> features, Vector<Float32x4> origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getError(prediction, origLabels);
  }

  Vector<Float32x4> predict(Matrix<Float32x4, Vector<Float32x4>> features) {
    final labels = List<double>(features.rowsNum);
    for (int i = 0; i < features.rowsNum; i++) {
      labels[i] = _weights.dot(features.getRowVector(i));
    }
    return Float32x4VectorFactory.from(labels);
  }
}
