import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';

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
