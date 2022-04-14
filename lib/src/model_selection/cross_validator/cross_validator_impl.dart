import 'package:ml_algo/src/common/exception/invalid_test_data_columns_number_exception.dart';
import 'package:ml_algo/src/common/exception/invalid_train_data_columns_number_exception.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

class CrossValidatorImpl implements CrossValidator {
  CrossValidatorImpl(
    this.samples,
    this._splitter,
    this.dtype,
  );

  final DataFrame samples;
  final DType dtype;
  final SplitIndicesProvider _splitter;

  @override
  Future<Vector> evaluate(
    ModelFactory predictorFactory,
    MetricType metricType, {
    DataPreprocessFn? onDataSplit,
  }) {
    final samplesAsMatrix = samples.toMatrix(dtype);
    final sourceColumnsNum = samplesAsMatrix.columnsNum;
    final discreteColumns = enumerate(samples.series)
        .where((indexedSeries) => indexedSeries.value.isDiscrete)
        .map((indexedSeries) => indexedSeries.index);
    final allIndicesGroups = _splitter.getIndices(samplesAsMatrix.rowsNum);
    final scores = allIndicesGroups.map((testRowsIndices) {
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
        throw InvalidTrainDataColumnsNumberException(
            sourceColumnsNum, transformedTrainDataColumnsNum);
      }

      if (transformedTestDataColumnsNum != sourceColumnsNum) {
        throw InvalidTestDataColumnsNumberException(
            sourceColumnsNum, transformedTestDataColumnsNum);
      }

      return predictorFactory(transformedTrainData)
          .assess(transformedTestData, metricType);
    }).toList();

    return Future.value(Vector.fromList(scores, dtype: dtype));
  }

  List<DataFrame> _makeSplit(
      Iterable<int> testRowsIndices, Iterable<int> discreteColumns) {
    final samplesAsMatrix = samples.toMatrix(dtype);
    final testRowsIndicesAsSet = Set<int>.from(testRowsIndices);
    final trainSamples = List<Vector>.filled(
      samplesAsMatrix.rowsNum - testRowsIndicesAsSet.length,
      Vector.empty(dtype: dtype),
    );
    final testSamples = List<Vector>.filled(
      testRowsIndicesAsSet.length,
      Vector.empty(dtype: dtype),
    );

    var trainSamplesCounter = 0;
    var testSamplesCounter = 0;

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
