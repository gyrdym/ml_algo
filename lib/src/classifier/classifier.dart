import 'package:dart_ml/src/data_preprocessing/intercept_preprocessor.dart';
import 'package:dart_ml/src/score_to_prob_link_function/link_function.dart' as scoreToProbLink;
import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

abstract class Classifier implements Evaluable {

  final Optimizer _optimizer;
  final _weightsByClasses = <Float32x4Vector>[];
  final scoreToProbLink.ScoreToProbLinkFunction _linkScoreToProbability;
  final InterceptPreprocessor _interceptPreprocessor;

  List<Float32x4Vector> get weightsByClasses => _weightsByClasses;

  Float32x4Vector get classLabels => _classLabels;
  Float32x4Vector _classLabels;

  Classifier(this._optimizer, this._linkScoreToProbability, double interceptScale) :
    _interceptPreprocessor = new InterceptPreprocessor(interceptScale: interceptScale);

  @override
  void fit(
    covariant List<Float32x4Vector> features,
    covariant Float32x4Vector origLabels,
    {
      covariant Float32x4Vector initialWeights,
      bool isDataNormalized = false
    }
  ) {
    _classLabels = origLabels.unique();

    final _features = _interceptPreprocessor.addIntercept(features);

    for (final targetLabel in _classLabels) {
      final labels = _makeLabelsOneVsAll(origLabels, targetLabel);
      _weightsByClasses.add(_optimizer.findExtrema(_features, labels,
        initialWeights: initialWeights,
        arePointsNormalized: isDataNormalized,
        isMinimizingObjective: false
      ));
    }
  }

  Float32x4Vector _makeLabelsOneVsAll(Float32x4Vector origLabels, double targetLabel) =>
    new Float32x4Vector.from(origLabels.map((double label) => label == targetLabel ? 1.0 : 0.0));

  List<Float32x4Vector> _addIntercept(List<Float32x4Vector> points) {
    final _points = new List<Float32x4Vector>(points.length);
    for (int i = 0; i < points.length; i++) {
      _points[i] = new Float32x4Vector.from(
          points[i].toList(growable: true)
            ..insert(0, 1.0)
          );
    }
    return _points;
  }

  @override
  double test(
    covariant List<Float32x4Vector> features,
    covariant Float32x4Vector origLabels,
    MetricType metricType
  ) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predictClasses(features);
    return metric.getError(prediction, origLabels);
  }

  List<Float32x4Vector> predictProbabilities(List<Float32x4Vector> features, {interceptConsidered: false}) {
    final _features = !interceptConsidered ? _interceptPreprocessor.addIntercept(features) : features;
    final distributions = new List<Float32x4Vector>(_features.length);

    for (int i = 0; i < _features.length; i++) {
      final probabilities = new List<double>(_weightsByClasses.length);
      for (int i = 0; i < _weightsByClasses.length; i++) {
        final score = _weightsByClasses[i].dot(_features[i]);
        probabilities[i] = _linkScoreToProbability(score);
      }
      distributions[i] = new Float32x4Vector.from(probabilities);
    }
    return distributions;
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features, {interceptConsidered: false}) {
    final _features = !interceptConsidered ? _interceptPreprocessor.addIntercept(features) : features;
    final distributions = predictProbabilities(_features, interceptConsidered: true);
    final classes = new List<double>(_features.length);
    for (int i = 0; i < distributions.length; i++) {
      final probabilities = distributions[i];
      classes[i] = probabilities.indexOf(probabilities.max()) * 1.0;
    }
    return new Float32x4Vector.from(classes);
  }
}
