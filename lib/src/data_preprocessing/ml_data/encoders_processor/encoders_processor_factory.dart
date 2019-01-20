import 'package:logging/logging.dart';
import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/encoders_processor/encoders_processor.dart';

abstract class MLDataEncodersProcessorFactory {
  MLDataEncodersProcessor create(List<List<Object>> data, List<String> header,
      CategoricalDataEncoderFactory encoderFactory, CategoricalDataEncoderType fallbackEncoderType, Logger logger);
}
