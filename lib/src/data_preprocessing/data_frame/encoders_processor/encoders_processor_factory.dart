import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor.dart';

abstract class EncodersProcessorFactory {
  EncodersProcessor create(
      List<String> header,
      CategoricalDataEncoderFactory encoderFactory,
      [Type dtype]
  );
}
