import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

class CrossValidatorImpl implements CrossValidator {
  CrossValidatorImpl(this.samples, this.targetNames, this._splitter, this.dtype);

  final DataFrame samples;
  final DType dtype;
  final Iterable<String> targetNames;
  final Splitter _splitter;

  @override
  double evaluate(PredictorFactory predictorFactory, MetricType metricType, {
    DataPreprocessFn dataPreprocessFn,
  }) {
    final samplesAsMatrix = samples.toMatrix();
    final discreteColumns = enumerate(samples.series)
        .where((indexedSeries) => indexedSeries.value.isDiscrete)
        .map((indexedSeries) => indexedSeries.index);

    final allIndicesGroups = _splitter.split(samplesAsMatrix.rowsNum);
    var score = 0.0;
    var folds = 0;

    for (final testRowsIndices in allIndicesGroups) {
      final testRowsIndicesAsSet = Set<int>.from(testRowsIndices);
      final trainSamples =
          List<Vector>(samplesAsMatrix.rowsNum - testRowsIndicesAsSet.length);
      final testSamples = List<Vector>(testRowsIndicesAsSet.length);

      int trainPointsCounter = 0;
      int testPointsCounter = 0;

      samplesAsMatrix.rowIndices.forEach((i) {
        if (testRowsIndicesAsSet.contains(i)) {
          testSamples[testPointsCounter++] = samplesAsMatrix[i];
        } else {
          trainSamples[trainPointsCounter++] = samplesAsMatrix[i];
        }
      });

      final trainingDataFrame = DataFrame.fromMatrix(
        Matrix.fromRows(trainSamples),
        header: samples.header,
        discreteColumns: discreteColumns,
      );

      final testingDataFrame = DataFrame.fromMatrix(
        Matrix.fromRows(testSamples),
        header: samples.header,
        discreteColumns: discreteColumns,
      );

      final splits = dataPreprocessFn != null
          ? dataPreprocessFn(trainingDataFrame, testingDataFrame)
          : [trainingDataFrame, testingDataFrame];

      final predictor = predictorFactory(splits[0], targetNames);

      score += predictor.assess(splits[1], targetNames, metricType);

      folds++;
    }

    return score / folds;
  }
}
