import 'package:dart_ml/src/data_splitter/k_fold.dart';
import 'package:dart_ml/src/data_splitter/leave_p_out.dart';
import 'package:dart_ml/src/data_splitter/splitter.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:simd_vector/vector.dart' show Vector;

class CrossValidator<T extends Vector> {
  final Splitter _splitter;

  factory CrossValidator.KFold({int numberOfFolds = 5}) =>
      new CrossValidator._(new KFoldSplitter(numberOfFolds));

  factory CrossValidator.LPO({int p = 5}) =>
      new CrossValidator._(new LeavePOutSplitter(p));

  CrossValidator._(this._splitter);

  double evaluate(
    Evaluable predictor,
    List<T> points,
    T labels,
    MetricType metric,
    {bool isDataNormalized = false}
  ) {

    if (points.length != labels.length) {
      throw new Exception('Number of feature objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(points.length);
    final scores = new List<double>(allIndicesGroups.length);
    int scoreCounter = 0;

    for (final testIndices in allIndicesGroups) {
      final trainFeatures = new List<T>(points.length - testIndices.length);
      final testFeatures = new List<T>(testIndices.length);
      final trainIndices = new List<int>(points.length - testIndices.length);

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
