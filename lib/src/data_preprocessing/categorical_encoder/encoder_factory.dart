import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';

class CategoricalDataEncoderFactory {
  const CategoricalDataEncoderFactory();

  CategoricalDataEncoder fromType(CategoricalDataEncoderType encoderType,
      [List<Object> categories]) {
    switch (encoderType) {
      case CategoricalDataEncoderType.oneHot:
        return oneHot();
      case CategoricalDataEncoderType.ordinal:
        return ordinal();
      default:
        throw Exception('Unknown categorical data encoder type has been '
            'provided');
    }
  }

  CategoricalDataEncoder oneHot() => OneHotEncoder();

  CategoricalDataEncoder ordinal() => OrdinalEncoder();
}
