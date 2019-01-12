import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';

CategoricalDataEncoder createEncoder({
  EncodeUnknownValueStrategy strategy,
  CategoryValuesExtractor extractor,
  List<Object> values,
  CategoricalDataEncoderType type = CategoricalDataEncoderType.oneHot,
}) {
  switch (type) {
    case CategoricalDataEncoderType.oneHot:
      return OneHotEncoder(encodeUnknownValueStrategy: strategy, valuesExtractor: extractor)..setCategoryValues(values);
    case CategoricalDataEncoderType.ordinal:
      return OrdinalEncoder(encodeUnknownValueStrategy: strategy, valuesExtractor: extractor)..setCategoryValues(values);
    default:
      throw Error();
  }
}
