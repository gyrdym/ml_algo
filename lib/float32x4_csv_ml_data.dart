import 'dart:typed_data';

import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/csv_ml_data.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/ml_data.dart';
import 'package:tuple/tuple.dart';

/// An abstract factory for instantiating a [Float32x4CsvMLDataInternal] exemplar. The latter is used for reading
/// csv-files and getting distinct data structures for features and labels.
abstract class Float32x4CsvMLData implements MLData<Float32x4> {

  /// Creates a csv-data instance from file. Resulting instance uses [Float32x4] data type for features and labels
  /// representation
  /// [fileName] Target csv-file name
  /// [labelIdx] Position of the label column (by default - the last column). Required parameter.
  /// [eol] End of line symbol of the csv-file
  /// [headerExists] Indicates, whether the csv-file header (a sequence of column names) exists or not
  /// [categories] Category names and all the categories' possible values. It only makes sense if [headerExists] is
  /// true. Use it, if you know in advance all the categorical values. If [categories] or [categoriesByIndexes] are
  /// provided, data will be processed much faster
  /// [categoriesByIndexes] Categories' column indexes and all the categories' possible values. Use it, if you know in
  /// advance all the categorical values. If [categories] or [categoriesByIndexes] are provided, data will be processed
  /// much faster
  /// [categoriesByIndexes] Categories' column indexes and all their possible values
  /// [categoryToEncoder] A map, that links category name to the encoder type, which will be used to encode this column's
  /// values
  /// [categoryIndexToEncoder] A map, that links category's column index to the encoder type, which will be used to
  /// encode this column's values. It only makes sense if [headerExists] is true
  /// [rows] Ranges of rows to be read from csv-file. Ranges represented as closed interval, that means that, e.g.
  /// `const Tuple2<int, int>(1, 1)` is a valid interval that contains only one value - `1`
  /// [columns] Ranges of columns to be read from csv-file. Ranges represented as closed interval, that means that, e.g.
  /// `const Tuple2<int, int>(1, 1)` is a valid interval that contains only one value - `1`
  /// [encoderType] Encoder for categorical data. Use it, if you want to encode all your categorical data with only
  /// this encoder. It makes sense only if [categoryToEncoder] is null
  /// [encodeUnknownStrategy] Indicates, what type of strategy will be used when unknown categorical value is
  /// encountered. E.g., if the strategy is [EncodeUnknownValueStrategy.returnZeroes], an unknown value just will be converted
  /// to the sequence of `0`
  factory Float32x4CsvMLData.fromFile(String fileName, {
    int labelIdx,
    String eol,
    bool headerExists,
    Map<String, List<Object>> categories,
    Map<int, List<Object>> categoriesByIndexes,
    Map<String, CategoricalDataEncoderType> categoryToEncoder,
    Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
    List<Tuple2<int, int>> rows,
    List<Tuple2<int, int>> columns,
    CategoricalDataEncoderType encoderType,
    EncodeUnknownValueStrategy encodeUnknownStrategy,
  }) = Float32x4CsvMLDataInternal.fromFile;
}
