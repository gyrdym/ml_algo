import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class CategoricalDataEncoderFactory {
  const CategoricalDataEncoderFactory();

  CategoricalDataEncoder fromType(CategoricalDataEncoderType encoderType, Map<String, List<Object>> categories,
      [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) =>
      CategoricalDataEncoder.fromType(encoderType, categories, encodeUnknownValueStrategy);

  CategoricalDataEncoder oneHot(Map<String, List<Object>> categories,
      [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) =>
      CategoricalDataEncoder.oneHot(categories, encodeUnknownValueStrategy);

  CategoricalDataEncoder ordinal(Map<String, List<Object>> categories,
      [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) =>
      CategoricalDataEncoder.ordinal(categories, encodeUnknownValueStrategy);
}
