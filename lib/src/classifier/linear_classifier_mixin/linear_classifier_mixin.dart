import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/predictor/weights_finder.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

mixin LinearClassifierMixin implements LinearClassifier, WeightsFinder {
  Type get dtype;
  Optimizer get optimizer;
  InterceptPreprocessor get interceptPreprocessor;
  ScoreToProbMapper get scoreToProbMapper;

  @override
  MLVector get weights => null;

  @override
  MLMatrix get weightsByClasses => _weightsByClasses;
  MLMatrix _weightsByClasses;

  @override
  MLMatrix get classLabels => _classLabels;
  MLMatrix _classLabels;

  @override
  void fit(MLMatrix features, MLMatrix labels,
      {MLMatrix initialWeights, bool isDataNormalized = false}) {
    _classLabels = labels.uniqueRows();
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    _weightsByClasses = learnWeights(
        processedFeatures, labels, initialWeights, isDataNormalized);
  }

  @override
  double test(MLMatrix features, MLMatrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    return metric.getScore(predictClasses(features), origLabels);
  }

  @override
  MLMatrix predictProbabilities(MLMatrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures);
  }

  @override
  MLMatrix predictClasses(MLMatrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures).mapRows((probabilities) {
      final labelIdx = probabilities.toList().indexOf(probabilities.max());
      return _classLabels.getRow(labelIdx);
    });
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
