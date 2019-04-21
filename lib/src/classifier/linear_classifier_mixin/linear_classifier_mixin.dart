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
  Matrix get weightsByClasses => _weightsByClasses;
  Matrix _weightsByClasses;

  @override
  void fit({Matrix initialWeights}) {
    _weightsByClasses = learnWeights(
        interceptPreprocessor.addIntercept(trainingFeatures), trainingOutcomes,
        initialWeights);
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    return metric.getScore(predictClasses(features), origLabels);
  }

  @override
  Matrix predictProbabilities(Matrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return checkDataAndPredictProbabilities(processedFeatures);
  }

  Matrix predictSingleClass(Matrix features, [double threshold = .5]) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    final classesSource = checkDataAndPredictProbabilities(processedFeatures)
        .getColumn(0)
        // TODO: use SIMD
        .map((value) => value >= threshold ? 1.0 : 0.0);
    return Matrix.fromColumns([Vector.from(classesSource)]);
  }

  Matrix predictMultiClass(Matrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return checkDataAndPredictProbabilities(processedFeatures)
        .mapRows((probabilities) {
      final labelIdx = probabilities.toList().indexOf(probabilities.max());
      return classLabels.getRow(labelIdx);
    });
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
