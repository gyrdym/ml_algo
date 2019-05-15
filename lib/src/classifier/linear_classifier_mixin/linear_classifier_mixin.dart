import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/helpers/add_intercept.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/matrix.dart';

mixin LinearClassifierMixin implements LinearClassifier {
  Optimizer get optimizer;
  ScoreToProbMapper get scoreToProbMapper;

  @override
  Matrix get weightsByClasses => _weightsByClasses;
  Matrix _weightsByClasses;

  @override
  void fit({Matrix initialWeights}) {
    _weightsByClasses ??= optimizer.findExtrema(initialWeights: initialWeights,
        isMinimizingObjective: false);
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    return metric.getScore(predictClasses(features), origLabels);
  }

  @override
  Matrix predictProbabilities(Matrix features) {
    final processedFeatures = addInterceptIf(trainingFeatures, fitIntercept,
        interceptScale);
    return checkDataAndPredictProbabilities(processedFeatures);
  }

  Matrix checkDataAndPredictProbabilities(Matrix features) {
    if (features.columnsNum != _weightsByClasses.rowsNum) {
      throw Exception('Wrong features number provided: expected '
          '${_weightsByClasses.rowsNum}, but ${features.columnsNum} given. '
          'Please, recheck columns number of the passed feature matrix');
    }
    return scoreToProbMapper.getProbabilities(features * _weightsByClasses);
  }
}
