import 'dart:typed_data';

import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/type.dart';
import 'package:ml_algo/predictor.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart' as scoreToProbLink;
import 'package:ml_linalg/linalg.dart';

abstract class Float32x4LinearClassifier implements Predictor<Float32x4> {
  final Optimizer<Float32x4> _optimizer;
  final scoreToProbLink.VectorizedScoreToProbLinkFunction<Float32x4> _linkScoreToProbability;
  final InterceptPreprocessor _interceptPreprocessor;
  final _vectorizedZero = Float32x4.zero();
  final _vectorizedOne = Float32x4.splat(1.0);
  final _classesMap = Map<double, Float32x4>();

  Float32x4LinearClassifier(this._optimizer, this._linkScoreToProbability, double interceptScale)
      : _interceptPreprocessor = InterceptPreprocessor(interceptScale: interceptScale);

  MLMatrix<Float32x4> get weightsByClasses => _weightsByClasses;
  MLMatrix<Float32x4> _weightsByClasses;

  MLVector<Float32x4> get classLabels => _classLabels;
  MLVector<Float32x4> _classLabels;

  @override
  void fit(MLMatrix<Float32x4> features, MLVector<Float32x4> origLabels,
      {MLVector<Float32x4> initialWeights, bool isDataNormalized = false}) {
    _classLabels = origLabels.unique();
    final labelsAsList = _classLabels.toList();
    final processedFeatures = _interceptPreprocessor.addIntercept(features);
    final weights = List<MLVector<Float32x4>>.generate(labelsAsList.length, (int i) {
      final labels = _makeLabelsOneVsAll(origLabels, labelsAsList[i]);
      return _optimizer.findExtrema(processedFeatures, labels,
          initialWeights: initialWeights, arePointsNormalized: isDataNormalized, isMinimizingObjective: false);
    });
    _weightsByClasses = Float32x4Matrix.columns(weights);
  }

  MLVector<Float32x4> _makeLabelsOneVsAll(MLVector<Float32x4> origLabels, double targetLabel) {
    _classesMap.putIfAbsent(targetLabel, () => Float32x4.splat(targetLabel));
    final target = _classesMap[targetLabel];
    return origLabels
        .vectorizedMap((Float32x4 element, [int start, int end]) => element.equal(target)
        .select(_vectorizedOne, _vectorizedZero));
  }

  @override
  double test(MLMatrix<Float32x4> features, MLVector<Float32x4> origLabels, MetricType metricType) {
    final evaluator = MetricFactory.createByType(metricType);
    final prediction = predictClasses(features);
    return evaluator.getError(prediction, origLabels);
  }

  MLMatrix<Float32x4> predictProbabilities(MLMatrix<Float32x4> features) {
    final processedFeatures = _interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures);
  }

  MLVector<Float32x4> predictClasses(MLMatrix<Float32x4> features) {
    final processedFeatures = _interceptPreprocessor.addIntercept(features);
    final distributions = _predictProbabilities(processedFeatures);
    final classes = List<double>(processedFeatures.rowsNum);
    for (int i = 0; i < distributions.rowsNum; i++) {
      final probabilities = distributions.getRow(i);
      classes[i] = probabilities.toList().indexOf(probabilities.max()) * 1.0;
    }
    return Float32x4Vector.from(classes);
  }

  MLMatrix<Float32x4> _predictProbabilities(MLMatrix<Float32x4> processedFeatures) {
    final distributions = List<MLVector<Float32x4>>(_weightsByClasses.columnsNum);
    for (int i = 0; i < _weightsByClasses.columnsNum; i++) {
      final scores = (processedFeatures * _weightsByClasses.getColumn(i)).toVector();
      distributions[i] = scores.vectorizedMap((Float32x4 el, [int startOffset, int endOffset]) =>
          _linkScoreToProbability(el));
    }
    return Float32x4Matrix.columns(distributions);
  }
}