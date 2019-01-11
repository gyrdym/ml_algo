import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';

/// A categorical data encoder. Contains names and values of the categories that supposed to be encoded and provides
/// method for data encoding
abstract class CategoricalDataEncoder {
  factory CategoricalDataEncoder.fromType(CategoricalDataEncoderType type,
      [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) {
    switch (type) {
      case CategoricalDataEncoderType.ordinal:
        return CategoricalDataEncoder.ordinal(encodeUnknownValueStrategy: encodeUnknownValueStrategy);
      case CategoricalDataEncoderType.oneHot:
        return CategoricalDataEncoder.oneHot(encodeUnknownValueStrategy: encodeUnknownValueStrategy);
      default:
        throw Error();
    }
  }

  factory CategoricalDataEncoder.oneHot({EncodeUnknownValueStrategy encodeUnknownValueStrategy}) = OneHotEncoder;

  factory CategoricalDataEncoder.ordinal({EncodeUnknownValueStrategy encodeUnknownValueStrategy}) = OrdinalEncoder;

  /// Specifies what to do with NaN(in numeric context)/null/empty or other unknown values for the particular category
  EncodeUnknownValueStrategy get encodeUnknownValueStrategy;

  /// Encodes passed categorical value to a numerical representation
  Iterable<double> encode(Object value);

  /// Find all unique values in the given list
  void setCategoryValues(List<Object> values);
}
