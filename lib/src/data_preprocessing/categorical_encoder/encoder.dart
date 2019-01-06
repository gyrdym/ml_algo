import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';

/// A categorical data encoder. Contains names and values of the categories that supposed to be encoded and provides
/// method for data encoding
abstract class CategoricalDataEncoder {
  factory CategoricalDataEncoder.fromType(CategoricalDataEncoderType type, Map<String, List<Object>> categories,
      [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) {
    switch (type) {
      case CategoricalDataEncoderType.ordinal:
        return CategoricalDataEncoder.ordinal(categories, encodeUnknownValueStrategy);
      case CategoricalDataEncoderType.oneHot:
        return CategoricalDataEncoder.oneHot(categories, encodeUnknownValueStrategy);
      default:
        throw Error();
    }
  }

  factory CategoricalDataEncoder.oneHot(Map<String, List<Object>> categories,
      [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) = OneHotEncoder;

  factory CategoricalDataEncoder.ordinal(Map<String, List<Object>> categories,
      [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) = OrdinalEncoder;

  /// Target categories. The key of the map - a category name, the value - a collection of all possible category values
  Map<String, Iterable<Object>> get categories;

  /// Specifies what to do with NaN(in numeric context)/null/empty values
  EncodeUnknownValueStrategy get encodeUnknownValueStrategy;

  /// Encodes passed categorical value to numerical representation
  Iterable<double> encode(String categoryLabel, Object value);
}
