import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';

abstract class EncodersProcessor {
  Map<int, CategoricalDataEncoder> createEncoders(
      Map<int, CategoricalDataEncoderType> indexesToEncoderTypes,
      Map<String, CategoricalDataEncoderType> namesToEncoderTypes,
  );
}
