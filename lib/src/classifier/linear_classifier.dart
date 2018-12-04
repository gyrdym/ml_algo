import 'dart:typed_data';

import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart' as scoreToProbLink;
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/type.dart';
import 'package:ml_algo/src/model_selection/evaluable.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_linalg/linalg.dart';

abstract class LinearClassifier implements Evaluable<Float32x4, MLVector<Float32x4>> {
  final Optimizer<Float32x4, MLVector<Float32x4>> _optimizer;
  final _weightsByClasses = <MLVector<Float32x4>>[];
  final scoreToProbLink.ScoreToProbLinkFunction _linkScoreToProbability;
  final InterceptPreprocessor _interceptPreprocessor;

  LinearClassifier(this._optimizer, this._linkScoreToProbability, double interceptScale)
      : _interceptPreprocessor = InterceptPreprocessor(interceptScale: interceptScale);

  List<MLVector<Float32x4>> get weightsByClasses => _weightsByClasses;
  MLVector<Float32x4> get classLabels => _classLabels;
  MLVector<Float32x4> _classLabels;

  @override
  void fit(MLMatrix<Float32x4, MLVector<Float32x4>> features, MLVector<Float32x4> origLabels,
      {MLVector<Float32x4> initialWeights, bool isDataNormalized = false}) {
    _classLabels = origLabels.unique();
    final _features = _interceptPreprocessor.addIntercept(features);
    for (int i = 0; i < _classLabels.length; i++) {
      final targetLabel = _classLabels[i];
      final labels = _makeLabelsOneVsAll(origLabels, targetLabel);
      _weightsByClasses.add(_optimizer.findExtrema(_features, labels,
          initialWeights: initialWeights, arePointsNormalized: isDataNormalized, isMinimizingObjective: false));
    }
  }

  MLVector<Float32x4> _makeLabelsOneVsAll(MLVector<Float32x4> origLabels, double targetLabel) {
    final target = Float32x4.splat(targetLabel);
    final zero = Float32x4.zero();
    return origLabels.vectorizedMap((Float32x4 element) => element.equal(target).select(element, zero));
  }

  @override
  double test(
      MLMatrix<Float32x4, MLVector<Float32x4>> features, MLVector<Float32x4> origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predictClasses(features);
    return metric.getError(prediction, origLabels);
  }

  List<MLVector<Float32x4>> predictProbabilities(MLMatrix<Float32x4, MLVector<Float32x4>> features,
      {bool interceptConsidered = false}) {
    final _features = !interceptConsidered ? _interceptPreprocessor.addIntercept(features) : features;
    final distributions = List<MLVector<Float32x4>>(_features.rowsNum);
    for (int i = 0; i < _features.rowsNum; i++) {
      final probabilities = List<double>(_weightsByClasses.length);
      for (int i = 0; i < _weightsByClasses.length; i++) {
        final score = _weightsByClasses[i].dot(_features.getRowVector(i));
        probabilities[i] = _linkScoreToProbability(score);
      }
      distributions[i] = Float32x4VectorFactory.from(probabilities);
    }
    return distributions;
  }

  MLVector<Float32x4> predictClasses(MLMatrix<Float32x4, MLVector<Float32x4>> features,
      {bool interceptConsidered = false}) {
    final _features = interceptConsidered ? features : _interceptPreprocessor.addIntercept(features);
    final distributions = predictProbabilities(_features, interceptConsidered: true);
    final classes = List<double>(_features.rowsNum);
    for (int i = 0; i < distributions.length; i++) {
      final probabilities = distributions[i];
      classes[i] = probabilities.toList().indexOf(probabilities.max()) * 1.0;
    }
    return Float32x4VectorFactory.from(classes);
  }
}
