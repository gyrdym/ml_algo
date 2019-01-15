import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/encoders_processor/encoders_processor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/encoders_processor/encoders_processor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/encoders_processor/encoders_processor_impl.dart';

class MLDataEncodersProcessorFactoryImpl implements MLDataEncodersProcessorFactory {
  const MLDataEncodersProcessorFactoryImpl();

  @override
  MLDataEncodersProcessor create(List<String> header, CategoricalDataEncoderFactory encoderFactory,
      CategoricalDataEncoderType fallbackEncoderType) =>
      MLDataEncodersProcessorImpl(header, encoderFactory, fallbackEncoderType);
}
