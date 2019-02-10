import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';

class CategoricalDataEncoderFactory {
  const CategoricalDataEncoderFactory();

  CategoricalDataEncoder fromType(CategoricalDataEncoderType encoderType,
          [List<Object> categories,
          EncodeUnknownValueStrategy encodeUnknownValueStrategy]) =>
      CategoricalDataEncoder.fromType(encoderType, encodeUnknownValueStrategy);

  CategoricalDataEncoder oneHot(
          [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) =>
      CategoricalDataEncoder.oneHot(
          encodeUnknownValueStrategy: encodeUnknownValueStrategy);

  CategoricalDataEncoder ordinal(
          [EncodeUnknownValueStrategy encodeUnknownValueStrategy]) =>
      CategoricalDataEncoder.ordinal(
          encodeUnknownValueStrategy: encodeUnknownValueStrategy);
}
