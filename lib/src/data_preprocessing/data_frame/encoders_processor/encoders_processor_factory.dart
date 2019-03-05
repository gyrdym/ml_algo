import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor.dart';

abstract class DataFrameEncodersProcessorFactory {
  DataFrameEncodersProcessor create(
      List<List<Object>> data,
      List<String> header,
      CategoricalDataEncoderFactory encoderFactory,
      CategoricalDataEncoderType fallbackEncoderType,
      Logger logger);
}
