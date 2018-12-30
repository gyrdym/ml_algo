import 'dart:typed_data';

import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/csv_ml_data.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/ml_data.dart';
import 'package:tuple/tuple.dart';

/// An abstract factory for instantiating a [Float32x4CsvMLDataInternal] exemplar. The latter is used for reading
/// csv-files and getting distinct data structures for features and labels.
abstract class Float32x4CsvMLData {

  /// Creates a csv-data instance from file. Resulting instance uses [Float32x4] data type for features and labels
  /// representation
  /// [fileName] Target csv-file name
  /// [labelIdx] Position of the label column (by default - the last column)
  /// [eol] End of line symbol of the csv-file
  /// [headerExists] Indicates, whether the csv-file header exists or not
  /// [categories] Categorical data labels and its possible values
  /// [columns] Ranges of columns to be read from csv-file
  /// [encoderType] Encoder for categorical data
  static MLData<Float32x4> fromFile(String fileName, int labelIdx, {
    String eol = '\n',
    bool headerExists = true,
    Map<String, List<Object>> categories,
    List<Tuple2<int, int>> columns,
    CategoricalDataEncoderType encoderType = CategoricalDataEncoderType.oneHot,
  }) =>
      Float32x4CsvMLDataInternal.fromFile(fileName,
          labelIdx: labelIdx,
          eol: eol,
          headerExists: headerExists,
          categories: categories,
          columns: columns,
          encoderType: encoderType,
      );
}