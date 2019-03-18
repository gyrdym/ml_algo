import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';

CategoricalDataEncoder createEncoder({
  EncodeUnknownValueStrategy strategy,
  CategoryValuesExtractor extractor,
  List<String> values,
  CategoricalDataEncoderType type = CategoricalDataEncoderType.oneHot,
}) {
  switch (type) {
    case CategoricalDataEncoderType.oneHot:
      return OneHotEncoder(
          encodeUnknownValueStrategy: strategy, valuesExtractor: extractor,
          {categoryLabels: values});
    case CategoricalDataEncoderType.ordinal:
      return OrdinalEncoder(
          encodeUnknownValueStrategy: strategy, valuesExtractor: extractor,
          {categoryLabels: values});
    default:
      throw Error();
  }
}
