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
    Matrix<E, Vector<E>> points,
    Vector<E> labels,
    MetricType metric,
    {bool isDataNormalized = false}
  ) {

    if (points.rowsNum != labels.length) {
      throw Exception('Number of feature objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(points.rowsNum);
    final scores = List<double>(allIndicesGroups.length);
    int scoreCounter = 0;

    for (final testIndices in allIndicesGroups) {
      final trainFeatures = List<List<double>>(points.rowsNum - testIndices.length);
      final testFeatures = List<List<double>>(testIndices.length);
      final trainIndices = List<int>(points.rowsNum - testIndices.length);

      int trainPointsCounter = 0;
      int testPointsCounter = 0;

      for (int index = 0; index < points.rowsNum; index++) {
        if (testIndices.contains(index)) {
          testFeatures[testPointsCounter++] = points[index].toList();
        } else {
          trainIndices[trainPointsCounter] = index;
          trainFeatures[trainPointsCounter] = points[index].toList();
          trainPointsCounter++;
        }
      }

      predictor.fit(
          Float32x4MatrixFactory.from(trainFeatures),
          labels.query(trainIndices),
          isDataNormalized: isDataNormalized);

      scores[scoreCounter++] = predictor.test(
          Float32x4MatrixFactory.from(testFeatures),
          labels.query(testIndices), metric);
    }

    return scores.reduce((sum, value) => (sum ?? 0.0) + value) / scores.length;
  }
}
