import 'dart:async';

import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/csv_data_frame.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:tuple/tuple.dart';

/// A parser for raw data
abstract class DataFrame {
  /**
   * Creates a csv-data instance from file.
   *
   * [fileName] Target csv-file name
   *
   * [labelIdx] Position of the label column (by default - the last column).
   * Required parameter.
   *
   * [labelName] Name of a column, where label values are contained
   *
   * [eol] End of line symbol of the csv-file
   *
   * [headerExists] Indicates, whether the csv-file header (a sequence of
   * column names) exists or not
   *
   * [categoriesByIndexes] Categories' column indexes and all the categories'
   * possible values. Use it, if you know in advance all the categorical values.
   * If [categories] or [categoriesByIndexes] are provided, data will be
   * processed much faster
   *
   * [categoriesByIndexes] Categories' column indexes and all their possible
   * values
   *
   * [categoryNameToEncoder] A map, that links category name to the encoder
   * type, which will be used to encode this column's values
   *
   * [categoryIndexToEncoder] A map, that links category's column index to the
   * encoder type, which will be used to encode this column's values. It only
   * makes sense if [headerExists] is true
   *
   * [rows] Ranges of rows to be read from csv-file. Ranges represented as
   * closed interval, that means that, e.g. `const Tuple2<int, int>(1, 1)` is a
   * valid interval that contains only one value - `1`
   *
   * [columns] Ranges of columns to be read from csv-file. Ranges represented
   * as closed interval, that means that, e.g. `const Tuple2<int, int>(1, 1)`
   * is a valid interval that contains only one value - `1`
   */
  factory DataFrame.fromCsv(
    String fileName, {
    String eol,
    int labelIdx,
    String labelName,
    bool headerExists,
    String fieldDelimiter,
    Map<int, List<String>> categoriesByIndexes,
    Map<String, CategoricalDataEncoderType> categoryNameToEncoder,
    Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
    List<Tuple2<int, int>> rows,
    List<Tuple2<int, int>> columns,
    Type dtype,
  }) = CsvDataFrame.fromFile;

  /// A data structure, containing just dataset column headers (generally, first
  /// row of a dataset).
  ///
  /// It may be omitted (in this case `null` will be returned)
  Future<List<String>> get header;

  /// Processed and ready to use (by machine learning algorithms) dataset
  /// features.
  ///
  /// Keep in mind, that the number of columns of the feature matrix
  /// may differ from the number of elements in [header] because of categorical
  /// data, that might present in the source dataset
  Future<Matrix> get features;

  /// Processed and ready to use (by machine learning algorithms) dataset
  /// labels (Target values, e.g. class labels or regression values)
  Future<Matrix> get labels;

  /// Decodes given categorical encoded column
  ///
  /// [column] - a matrix, where each row is an encoded categorical value, e.g.
  /// with one-hot encoder
  ///
  /// [colName] - a name of encoded column
  ///
  /// [colIdx] - an index of encoded column
  Iterable<String> decode(Matrix column, {String colName, int colIdx});
}
