import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor_impl.dart';

class DataFrameEncodersProcessorFactoryImpl
    implements DataFrameEncodersProcessorFactory {
  const DataFrameEncodersProcessorFactoryImpl();

  @override
  DataFrameEncodersProcessor create(
          List<List<Object>> records,
          List<String> header,
          CategoricalDataEncoderFactory encoderFactory,
          CategoricalDataEncoderType fallbackEncoderType,
          Logger logger) =>
      DataFrameEncodersProcessorImpl(
          records, header, encoderFactory, fallbackEncoderType, logger);
}
