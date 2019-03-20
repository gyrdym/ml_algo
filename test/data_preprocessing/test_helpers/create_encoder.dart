import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';

CategoricalDataEncoder createEncoder({
  List<String> values,
  CategoricalDataEncoderType type = CategoricalDataEncoderType.oneHot,
}) {
  switch (type) {
    case CategoricalDataEncoderType.oneHot:
      return OneHotEncoder();
    case CategoricalDataEncoderType.ordinal:
      return OrdinalEncoder();
    default:
      throw Error();
  }
}
