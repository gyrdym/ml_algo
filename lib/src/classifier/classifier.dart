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

  List<Float32x4Vector> get weightsByClasses => _weightsByClasses;

  Float32x4Vector get classLabels => _classLabels;
  Float32x4Vector _classLabels;

  Classifier(this._optimizer, this._linkScoreToProbability);

  @override
  void fit(
    covariant List<Float32x4Vector> features,
    covariant Float32x4Vector origLabels,
    {
      covariant Float32x4Vector initialWeights,
      bool isDataNormalized
    }
  ) {
    _classLabels = origLabels.unique();
    for (final classLabel in _classLabels) {
      final labels = _makeLabelsOneVsAll(origLabels, classLabel);
      _weightsByClasses.add(_optimizer.findExtrema(features, labels,
        initialWeights: initialWeights,
        arePointsNormalized: isDataNormalized
      ));
    }
  }

  Float32x4Vector _makeLabelsOneVsAll(Float32x4Vector origLabels, double targetLabel) =>
    new Float32x4Vector.from(origLabels.map((double label) => label == targetLabel ? 1.0 : 0.0));

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

  List<Float32x4Vector> predictProbabilities(List<Float32x4Vector> points) {
    final distributions = new List<Float32x4Vector>(points.length);

    for (int i = 0; i < points.length; i++) {
      final probabilities = new List<double>(_weightsByClasses.length);
      for (int i = 0; i < _weightsByClasses.length; i++) {
        final score = _weightsByClasses[i].dot(points[i]);
        probabilities[i] = _linkScoreToProbability(score);
      }
      distributions[i] = new Float32x4Vector.from(probabilities);
    }
    return distributions;
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    final distributions = predictProbabilities(features);
    final classes = new List<double>(features.length);
    for (int i = 0; i < distributions.length; i++) {
      final probabilities = distributions[i];
      classes[i] = probabilities.indexOf(probabilities.max()) * 1.0;
    }
    return new Float32x4Vector.from(classes);
  }
}
