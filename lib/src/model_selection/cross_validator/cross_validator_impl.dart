import 'package:ml_algo/cross_validator.dart';
import 'package:ml_algo/predictor.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/data_splitter/k_fold.dart';
import 'package:ml_algo/src/model_selection/data_splitter/leave_p_out.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class CrossValidatorImpl implements CrossValidator {
  final Type dtype;
  final Splitter _splitter;

  factory CrossValidatorImpl.kFold({Type dtype, int numberOfFolds = 5}) =>
      CrossValidatorImpl._(dtype, KFoldSplitter(numberOfFolds));

  factory CrossValidatorImpl.lpo({Type dtype, int p}) =>
      CrossValidatorImpl._(dtype, LeavePOutSplitter(p));

  CrossValidatorImpl._(Type dtype, this._splitter)
      : dtype = dtype ?? DefaultParameterValues.dtype;

  @override
  double evaluate(
      Predictor predictor, MLMatrix points, MLVector labels, MetricType metric,
      {bool isDataNormalized = false}) {
    if (points.rowsNum != labels.length) {
      throw Exception(
          'Number of feature objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(points.rowsNum);
    final scores = List<double>(allIndicesGroups.length);
    int scoreCounter = 0;

    for (final testIndices in allIndicesGroups) {
      final trainFeatures =
          List<List<double>>(points.rowsNum - testIndices.length);
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

      predictor.fit(MLMatrix.from(trainFeatures, dtype: dtype),
          labels.query(trainIndices),
          isDataNormalized: isDataNormalized);

      scores[scoreCounter++] = predictor.test(
          MLMatrix.from(testFeatures, dtype: dtype),
          labels.query(testIndices),
          metric);
    }

    return scores.reduce((sum, value) => (sum ?? 0.0) + value) / scores.length;
  }
}
