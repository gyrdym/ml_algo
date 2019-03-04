import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/data_splitter/k_fold.dart';
import 'package:ml_algo/src/model_selection/data_splitter/leave_p_out.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class CrossValidatorImpl implements CrossValidator {
  factory CrossValidatorImpl.kFold({Type dtype, int numberOfFolds = 5}) =>
      CrossValidatorImpl._(dtype, KFoldSplitter(numberOfFolds));

  factory CrossValidatorImpl.lpo({Type dtype, int p}) =>
      CrossValidatorImpl._(dtype, LeavePOutSplitter(p));

  CrossValidatorImpl._(Type dtype, this._splitter)
      : dtype = dtype ?? DefaultParameterValues.dtype;

  final Type dtype;
  final Splitter _splitter;

  @override
  double evaluate(
      Predictor predictor, Matrix points, Matrix labels, MetricType metric,
      {bool isDataNormalized = false}) {
    if (points.rowsNum != labels.rowsNum) {
      throw Exception(
          'Number of feature objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(points.rowsNum);
    final scores = List<double>(allIndicesGroups.length);
    int scoreCounter = 0;

    for (final testIndices in allIndicesGroups) {
      final trainFeatures =
          List<Vector>(points.rowsNum - testIndices.length);
      final trainLabels =
          List<Vector>(points.rowsNum - testIndices.length);

      final testFeatures = List<Vector>(testIndices.length);
      final testLabels = List<Vector>(testIndices.length);

      int trainPointsCounter = 0;
      int testPointsCounter = 0;

      for (int index = 0; index < points.rowsNum; index++) {
        if (testIndices.contains(index)) {
          testFeatures[testPointsCounter] = points.getRow(index);
          testLabels[testPointsCounter] = labels.getRow(index);
          testPointsCounter++;
        } else {
          trainFeatures[trainPointsCounter] = points.getRow(index);
          trainLabels[trainPointsCounter] = labels.getRow(index);
          trainPointsCounter++;
        }
      }

      predictor.fit(
          Matrix.rows(trainFeatures, dtype: dtype),
          Matrix.rows(trainLabels, dtype: dtype),
          isDataNormalized: isDataNormalized);

      scores[scoreCounter++] = predictor.test(
          Matrix.rows(testFeatures, dtype: dtype),
          Matrix.rows(testLabels, dtype: dtype),
          metric
      );
    }

    return scores.reduce((sum, value) => (sum ?? 0.0) + value) / scores.length;
  }
}
