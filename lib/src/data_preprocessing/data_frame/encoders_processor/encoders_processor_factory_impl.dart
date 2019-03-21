import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor_impl.dart';

class EncodersProcessorFactoryImpl
    implements EncodersProcessorFactory {
  const EncodersProcessorFactoryImpl();

  @override
  EncodersProcessor create(
      List<List<Object>> records,
      List<String> header,
      CategoricalDataEncoderFactory encoderFactory
  ) => EncodersProcessorImpl(records, header, encoderFactory);
}
