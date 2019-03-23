import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';

abstract class CategoricalDataEncoderFactory {
  CategoricalDataEncoder fromType(CategoricalDataEncoderType encoderType,
      [Type dtype]);
  CategoricalDataEncoder oneHot([Type dtype]);
  CategoricalDataEncoder ordinal([Type dtype]);
}
