import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter.dart';
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
  final DataSplitter _splitter;

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
      final split = _makeSplit(testRowsIndices, discreteColumns);
      final trainDataFrame = split[0];
      final testDataFrame = split[1];

      final splits = onDataSplit != null
          ? onDataSplit(trainDataFrame, testDataFrame)
          : [trainDataFrame, testDataFrame];

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

      score += predictorFactory(transformedTrainData, targetNames)
          .assess(transformedTestData, targetNames, metricType);

      folds++;
    }

    return score / folds;
  }

  List<DataFrame> _makeSplit(Iterable<int> testRowsIndices,
      Iterable<int> discreteColumns) {
    final samplesAsMatrix = samples.toMatrix(dtype);
    final testRowsIndicesAsSet = Set<int>.from(testRowsIndices);
    final trainSamples =
      List<Vector>(samplesAsMatrix.rowsNum - testRowsIndicesAsSet.length);
    final testSamples = List<Vector>(testRowsIndicesAsSet.length);

    int trainSamplesCounter = 0;
    int testSamplesCounter = 0;

    samplesAsMatrix.rowIndices.forEach((i) {
      if (testRowsIndicesAsSet.contains(i)) {
        testSamples[testSamplesCounter++] = samplesAsMatrix[i];
      } else {
        trainSamples[trainSamplesCounter++] = samplesAsMatrix[i];
      }
    });

    return [
      DataFrame.fromMatrix(
        Matrix.fromRows(trainSamples, dtype: dtype),
        header: samples.header,
        discreteColumns: discreteColumns,
      ),
      DataFrame.fromMatrix(
        Matrix.fromRows(testSamples, dtype: dtype),
        header: samples.header,
        discreteColumns: discreteColumns,
      ),
    ];
  }
}
