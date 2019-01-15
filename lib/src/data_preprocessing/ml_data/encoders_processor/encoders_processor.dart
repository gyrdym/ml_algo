import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

abstract class MLDataEncodersProcessor {
  Map<int, CategoricalDataEncoder> createEncoders(Map<int, CategoricalDataEncoderType> indexesToEncoderTypes,
      Map<String, CategoricalDataEncoderType> namesToEncoderTypes, Map<String, List<Object>> categories);
}
