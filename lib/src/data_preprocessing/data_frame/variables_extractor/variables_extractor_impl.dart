import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/to_float_number_converter/to_float_number_converter.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:tuple/tuple.dart';

class VariablesExtractorImpl implements VariablesExtractor {
  VariablesExtractorImpl(this._observations, this._rowsMask, this._columnsMask,
      this._encoders, this._labelIdx, this._toFloatConverter,
      [Type dtype = DefaultParameterValues.dtype]) : _dtype = dtype {
    if (_columnsMask.length > _observations.first.length) {
      throw Exception(columnsMaskWrongLengthMsg);
    }
    if (_rowsMask.length > _observations.length) {
      throw Exception(rowsMaskWrongLengthMsg);
    }
  }

  static const String rowsMaskWrongLengthMsg =
      'Rows mask length should not be greater than actual rows number in the '
      'dataset!';

  static const String columnsMaskWrongLengthMsg =
      'Columns mask length should not be greater than actual columns number in '
      'the dataset!';

  final Type _dtype;
  final List<bool> _rowsMask;
  final List<bool> _columnsMask;
  final Map<int, CategoricalDataEncoder> _encoders;
  final int _labelIdx;
  final ToFloatNumberConverter _toFloatConverter;
  final List<List<Object>> _observations;

  Tuple2<Matrix, Matrix> _data;

  @override
  Matrix extractFeatures() => _extract().item1;

  @override
  Matrix extractLabels() => _extract().item2;

  Tuple2<Matrix, Matrix> _extract() {
    if (_data == null) {
      final columnsData = _collectColumnsData();
      _data = _processColumns(columnsData.item1, columnsData.item2);
    }
    return _data;
  }

  Tuple2<Map<int, List<double>>, Map<int, List<String>>> _collectColumnsData() {
    // key here is a zero-based number of column in the [records]
    final Map<int, List<String>> categoricalColumns = {};
    // key here is a zero-based number of column in the [records]
    final Map<int, List<double>> numericalColumns = {};

    for (int i = 0; i < _rowsMask.length; i++) {
      if (_rowsMask[i] == true) {
        final rowData = _processRow(_observations[i]);
        rowData.item1.forEach((idx, value) =>
            numericalColumns.putIfAbsent(idx, () => []).add(value));
        rowData.item2.forEach((idx, value) =>
            categoricalColumns.putIfAbsent(idx, () => []).add(value));
      }
    }
    return Tuple2(numericalColumns, categoricalColumns);
  }

  Tuple2<Map<int, double>, Map<int, String>> _processRow(
      List<Object> row) {
    final Map<int, double> numericalValues = {};
    final Map<int, String> categoricalValues = {};
    for (int i = 0; i < _columnsMask.length; i++) {
      if (_columnsMask[i] == false) {
        continue;
      }
      if (_encoders.containsKey(i)) {
        categoricalValues.putIfAbsent(i, () => row[i].toString());
      } else {
        numericalValues.putIfAbsent(i, () => _toFloatConverter.convert(row[i]));
      }
    }
    return Tuple2(numericalValues, categoricalValues);
  }

  Tuple2<Matrix, Matrix> _processColumns(
    Map<int, List<double>> numericalColumns,
    Map<int, List<String>> categoricalColumns,
  ) {
    final featureColumns = <Vector>[];
    final labelColumns = <Vector>[];
    final updateColumns = (int i, Vector vectorColumn) {
      i == _labelIdx
          ? labelColumns.add(vectorColumn)
          : featureColumns.add(vectorColumn);
    };
    for (int i = 0; i < _columnsMask.length; i++) {
      if (numericalColumns.containsKey(i)) {
        updateColumns(i, Vector.from(numericalColumns[i], dtype: _dtype));
      } else if (categoricalColumns.containsKey(i)) {
        final encoded = _encoders[i].encode(categoricalColumns[i]);
        for (int col = 0; col < encoded.columnsNum; col++) {
          updateColumns(i, encoded.getColumn(col));
        }
      }
    }
    return Tuple2(
        featureColumns.isNotEmpty
            ? Matrix.columns(featureColumns, dtype: _dtype) : null,
        labelColumns.isNotEmpty
            ? Matrix.columns(labelColumns, dtype: _dtype) : null
    );
  }
}
