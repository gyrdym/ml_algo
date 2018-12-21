import 'dart:typed_data';

import 'package:ml_algo/src/metric/type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/data_splitter/k_fold.dart';
import 'package:ml_algo/src/model_selection/data_splitter/leave_p_out.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_algo/src/model_selection/evaluable.dart';
import 'package:ml_linalg/linalg.dart';

class Float32x4CrossValidatorInternal implements CrossValidator<Float32x4> {
  final Splitter _splitter;

  factory Float32x4CrossValidatorInternal.kFold({int numberOfFolds = 5}) =>
      Float32x4CrossValidatorInternal._(KFoldSplitter(numberOfFolds));

  factory Float32x4CrossValidatorInternal.lpo({int p = 5}) => Float32x4CrossValidatorInternal._(LeavePOutSplitter(p));

  Float32x4CrossValidatorInternal._(this._splitter);

  @override
  double evaluate(Evaluable predictor, MLMatrix<Float32x4> points, MLVector<Float32x4> labels, MetricType metric,
      {bool isDataNormalized = false}) {
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

      predictor.fit(Float32x4Matrix.from(trainFeatures), labels.query(trainIndices),
          isDataNormalized: isDataNormalized);

      scores[scoreCounter++] =
          predictor.test(Float32x4Matrix.from(testFeatures), labels.query(testIndices), metric);
    }

    return scores.reduce((sum, value) => (sum ?? 0.0) + value) / scores.length;
  }
}
