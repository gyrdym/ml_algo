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
  Vector get weights => null;

  @override
  Matrix get weightsByClasses => _weightsByClasses;
  Matrix _weightsByClasses;

  @override
  Matrix get classLabels => _classLabels;
  Matrix _classLabels;

  @override
  void fit(Matrix features, Matrix labels,
      {Matrix initialWeights, bool isDataNormalized = false}) {
    _classLabels = labels.uniqueRows();
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    _weightsByClasses = learnWeights(
        processedFeatures, labels, initialWeights, isDataNormalized);
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    return metric.getScore(predictClasses(features), origLabels);
  }

  @override
  Matrix predictProbabilities(Matrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures);
  }

  @override
  Matrix predictClasses(Matrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures).mapRows((probabilities) {
      final labelIdx = probabilities.toList().indexOf(probabilities.max());
      return _classLabels.getRow(labelIdx);
    });
  }

  Matrix _predictProbabilities(Matrix features) {
    if (features.columnsNum != _weightsByClasses.rowsNum) {
      throw Exception('Wrong features number provided: expected '
          '${_weightsByClasses.rowsNum}, but ${features.columnsNum} given. '
          'Please, recheck columns number of the passed feature matrix');
    }
    return scoreToProbMapper.linkScoresToProbs(features * _weightsByClasses);
  }
}
