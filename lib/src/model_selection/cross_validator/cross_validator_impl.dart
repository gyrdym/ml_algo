import 'package:ml_algo/src/utils/default_parameter_values.dart';
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
  double evaluate(Predictor predictorFactory(Matrix features, Matrix outcomes),
      Matrix observations, Matrix labels, MetricType metric) {
    if (observations.rowsNum != labels.rowsNum) {
      throw Exception(
          'Number of feature objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(observations.rowsNum);
    var score = 0.0;
    var folds = 0;

    for (final testIndices in allIndicesGroups) {
      final trainFeatures =
          List<Vector>(observations.rowsNum - testIndices.length);
      final trainLabels =
          List<Vector>(observations.rowsNum - testIndices.length);

      final testFeatures = List<Vector>(testIndices.length);
      final testLabels = List<Vector>(testIndices.length);

      int trainPointsCounter = 0;
      int testPointsCounter = 0;

      for (int index = 0; index < observations.rowsNum; index++) {
        if (testIndices.contains(index)) {
          testFeatures[testPointsCounter] = observations.getRow(index);
          testLabels[testPointsCounter] = labels.getRow(index);
          testPointsCounter++;
        } else {
          trainFeatures[trainPointsCounter] = observations.getRow(index);
          trainLabels[trainPointsCounter] = labels.getRow(index);
          trainPointsCounter++;
        }
      }

      final predictor = predictorFactory(
        Matrix.fromRows(trainFeatures, dtype: dtype),
        Matrix.fromRows(trainLabels, dtype: dtype),
      )..fit();

      score += predictor.test(
          Matrix.fromRows(testFeatures, dtype: dtype),
          Matrix.fromRows(testLabels, dtype: dtype),
          metric
      );
      folds++;
    }

    return score / folds;
  }
}
