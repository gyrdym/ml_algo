import 'package:dart_ml/src/data_splitter/k_fold.dart';
import 'package:dart_ml/src/data_splitter/leave_p_out.dart';
import 'package:dart_ml/src/data_splitter/splitter.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:linalg/linalg.dart';

class CrossValidator<E> {

  final Splitter _splitter;

  factory CrossValidator.kFold({int numberOfFolds = 5}) =>
      CrossValidator._(KFoldSplitter(numberOfFolds));

  factory CrossValidator.lpo({int p = 5}) =>
      CrossValidator._(LeavePOutSplitter(p));

  CrossValidator._(this._splitter);

  double evaluate(
    Evaluable predictor,
    List<Vector<E>> points,
    Vector<E> labels,
    MetricType metric,
    {bool isDataNormalized = false}
  ) {

    if (points.length != labels.length) {
      throw Exception('Number of feature objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(points.length);
    final scores = List<double>(allIndicesGroups.length);
    int scoreCounter = 0;

    for (final testIndices in allIndicesGroups) {
      final trainFeatures = List<Vector<E>>(points.length - testIndices.length);
      final testFeatures = List<Vector<E>>(testIndices.length);
      final trainIndices = List<int>(points.length - testIndices.length);

      int trainPointsCounter = 0;
      int testPointsCounter = 0;

      for (int index = 0; index < points.length; index++) {
        if (testIndices.contains(index)) {
          testFeatures[testPointsCounter++] = points[index];
        } else {
          trainIndices[trainPointsCounter] = index;
          trainFeatures[trainPointsCounter] = points[index];
          trainPointsCounter++;
        }
      }

      predictor.fit(trainFeatures, labels.query(trainIndices), isDataNormalized: isDataNormalized);
      scores[scoreCounter++] = predictor.test(testFeatures, labels.query(testIndices), metric);
    }

    return scores.reduce((sum, value) => (sum ?? 0.0) + value) / scores.length;
  }
}
