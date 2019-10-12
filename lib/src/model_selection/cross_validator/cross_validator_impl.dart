import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

class CrossValidatorImpl implements CrossValidator {
  CrossValidatorImpl(
      this.samples,
      this.targetNames,
      this._splitter,
      this.dtype,
  );

  final DataFrame samples;
  final DType dtype;
  final Iterable<String> targetNames;
  final Splitter _splitter;

  @override
  double evaluate(PredictorFactory predictorFactory, MetricType metricType, {
    DataPreprocessFn onDataSplit,
  }) {
    final samplesAsMatrix = samples.toMatrix(dtype);
    final sourceColumnsNum = samplesAsMatrix.columnsNum;

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

      final splits = onDataSplit != null
          ? onDataSplit(trainingDataFrame, testingDataFrame)
          : [trainingDataFrame, testingDataFrame];

      final transformedTrainData = splits[0];
      final transformedTestData = splits[1];

      final transformedTrainDataColumnsNum = transformedTrainData.header.length;
      final transformedTestDataColumnsNum = transformedTestData.header.length;

      if (transformedTrainDataColumnsNum != sourceColumnsNum) {
        throw Exception('Unexpected columns number in training data: '
            'expected $sourceColumnsNum, received '
            '${transformedTrainDataColumnsNum}');
      }

      if (transformedTestDataColumnsNum != sourceColumnsNum) {
        throw Exception('Unexpected columns number in testing data: '
            'expected $sourceColumnsNum, received '
            '${transformedTestDataColumnsNum}');
      }

      final predictor = predictorFactory(transformedTrainData, targetNames);

      score += predictor.assess(transformedTestData, targetNames, metricType);

      folds++;
    }

    return score / folds;
  }
}
