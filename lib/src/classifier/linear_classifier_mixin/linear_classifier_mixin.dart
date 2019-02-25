import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/classifier/weights_finder/weights_finder.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

mixin LinearClassifierMixin implements LinearClassifier, WeightsFinder {
  Type dtype;
  Optimizer optimizer;
  InterceptPreprocessor interceptPreprocessor;
  LabelsProcessor labelsProcessor;
  ScoreToProbMapper scoreToProbMapper;

  @override
  MLVector get weights => null;

  @override
  MLMatrix get weightsByClasses => _weightsByClasses;
  MLMatrix _weightsByClasses;

  @override
  List<double> get classLabels => _classLabels;
  List<double> _classLabels;

  @override
  void fit(MLMatrix features, MLVector labels,
      {MLMatrix initialWeights, bool isDataNormalized = false}) {
    _classLabels = labels.unique().toList();
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    _weightsByClasses = learnWeights(
        processedFeatures, labels, initialWeights, isDataNormalized);
  }

  @override
  double test(MLMatrix features, MLVector origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    return metric.getScore(predictClasses(features), origLabels);
  }

  @override
  MLMatrix predictProbabilities(MLMatrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures);
  }

  @override
  MLVector predictClasses(MLMatrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    final distributions = _predictProbabilities(processedFeatures);
    final classes = List<double>(processedFeatures.rowsNum);
    for (int i = 0; i < distributions.rowsNum; i++) {
      final probabilities = distributions.getRow(i);
      classes[i] = probabilities.toList().indexOf(probabilities.max()) * 1.0;
    }
    return MLVector.from(classes, dtype: dtype);
  }

  MLMatrix _predictProbabilities(MLMatrix features) {
    if (features.columnsNum != _weightsByClasses.rowsNum) {
      throw Exception('Wrong features number provided: expected '
          '${_weightsByClasses.rowsNum}, but ${features.columnsNum} given. '
          'Please, recheck columns number of the passed feature matrix');
    }
    return scoreToProbMapper.linkScoresToProbs(features * _weightsByClasses);
  }
}
