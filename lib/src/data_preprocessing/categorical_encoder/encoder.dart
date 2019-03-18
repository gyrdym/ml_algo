import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';
import 'package:ml_linalg/matrix.dart';

/// A categorical data encoder. Contains names and values of the categories
/// that supposed to be encoded and provides method for data encoding
abstract class CategoricalDataEncoder {
  factory CategoricalDataEncoder.fromType(CategoricalDataEncoderType type) {
    switch (type) {
      case CategoricalDataEncoderType.ordinal:
        return CategoricalDataEncoder.ordinal();
      case CategoricalDataEncoderType.oneHot:
        return CategoricalDataEncoder.oneHot();
      default:
        throw Error();
    }
  }

  factory CategoricalDataEncoder.oneHot() = OneHotEncoder;

  factory CategoricalDataEncoder.ordinal() = OrdinalEncoder;

  /// Encodes passed categorical values to a numerical representation
  Matrix encode(Iterable<String> values);

  /// Decodes passed categorical encoded data to a source string representation
  Iterable<String> decode(Matrix values);
}
