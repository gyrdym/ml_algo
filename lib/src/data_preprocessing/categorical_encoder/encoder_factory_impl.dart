import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';

class CategoricalDataEncoderFactoryImpl implements
    CategoricalDataEncoderFactory {
  const CategoricalDataEncoderFactoryImpl();

  @override
  CategoricalDataEncoder fromType(CategoricalDataEncoderType encoderType,
      [Type dtype]) {
    switch (encoderType) {
      case CategoricalDataEncoderType.oneHot:
        return OneHotEncoder(dtype);
      case CategoricalDataEncoderType.ordinal:
        return OrdinalEncoder(dtype);
      default:
        throw Exception('Unknown categorical data encoder type has been '
            'provided');
    }
  }

  @override
  CategoricalDataEncoder oneHot([Type dtype]) => OneHotEncoder(dtype);

  @override
  CategoricalDataEncoder ordinal([Type dtype]) => OrdinalEncoder(dtype);
}
