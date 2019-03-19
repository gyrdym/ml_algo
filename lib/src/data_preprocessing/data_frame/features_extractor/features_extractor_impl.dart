import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:tuple/tuple.dart';

class DataFrameFeaturesExtractorImpl implements DataFrameFeaturesExtractor {

  DataFrameFeaturesExtractorImpl(this.records, this.rowsMask, this.columnsMask,
      this.encoders, this.labelIdx, this.toFloatConverter)
      : rowsNum = rowsMask.where((bool flag) => flag).length,
        columnsNum = columnsMask.where((bool flag) => flag).length {
    if (columnsMask.length > records.first.length) {
      throw Exception(columnsMaskWrongLengthMsg);
    }
    if (rowsMask.length > records.length) {
      throw Exception(rowsMaskWrongLengthMsg);
    }
  }

  static const String rowsMaskWrongLengthMsg =
      'Rows mask length should not be greater than actual rows number in the '
      'dataset!';

  static const String columnsMaskWrongLengthMsg =
      'Columns mask length should not be greater than actual columns number in '
      'the dataset!';

  final List<bool> rowsMask;
  final List<bool> columnsMask;
  final Map<int, CategoricalDataEncoder> encoders;
  final int rowsNum;
  final int columnsNum;
  final int labelIdx;
  final DataFrameValueConverter toFloatConverter;
  final List<List<Object>> records;

  @override
  Matrix extract() {
    final columnsData = _collectColumnsData();
    return _processColumns(columnsData.item1, columnsData.item2);
  }

  Tuple2<Map<int, List<double>>, Map<int, List<String>>> _collectColumnsData() {
    // key here is a zero-based number of column in the [records]
    final Map<int, List<String>> categoricalColumns = {};
    // key here is a zero-based number of column in the [records]
    final Map<int, List<double>> numericalColumns = {};

    for (int i = 0; i < rowsMask.length; i++) {
      if (rowsMask[i] == true) {
        final rowData = _processRow(records[i]);
        rowData.item1.forEach((idx, value) {
          numericalColumns.putIfAbsent(idx, () => [value]);
          numericalColumns[idx].add(value);
        });
        rowData.item2.forEach((idx, value) {
          categoricalColumns.putIfAbsent(idx, () => [value]);
          categoricalColumns[idx].add(value);
        });
      }
    }
    return Tuple2(numericalColumns, categoricalColumns);
  }

  Tuple2<Map<int, double>, Map<int, String>> _processRow(
      List<Object> row) {
    final Map<int, double> numericalValues = {};
    final Map<int, String> categoricalValues = {};
    for (int i = 0; i < columnsMask.length; i++) {
      if (labelIdx == i || columnsMask[i] == false) {
        continue;
      }
      if (encoders.containsKey(i)) {
        categoricalValues.putIfAbsent(i, () => row[i].toString());
      } else {
        numericalValues.putIfAbsent(i, () => toFloatConverter.convert(row[i]));
      }
    }
    return Tuple2(numericalValues, categoricalValues);
  }

  Matrix _processColumns(
    Map<int, List<double>> numericalColumns,
    Map<int, List<String>> categoricalColumns,
  ) {
    final columns = <Vector>[];
    for (int i = 0; i < columnsMask.length; i++) {
      if (numericalColumns.containsKey(i)) {
        columns.add(Vector.from(numericalColumns[i]));
      } else if (categoricalColumns.containsKey(i)) {
        final encoded = encoders[i].encode(categoricalColumns[i]);
        for (int col = 0; col < encoded.columnsNum; col++) {
          columns.add(encoded.getColumn(col));
        }
      }
    }
    return Matrix.columns(columns);
  }
}
