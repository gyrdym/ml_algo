import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';
import 'package:test/test.dart';

void main() {
  group('CategoricalDataEncoderFactory', () {
    test('should create one-hot encoder by type', () {
      final factory = const CategoricalDataEncoderFactoryImpl();
      final actual = factory.fromType(CategoricalDataEncoderType.oneHot);
      expect(actual.runtimeType, OneHotEncoder);
    });

    test('should create ordinal encoder by type', () {
      final factory = const CategoricalDataEncoderFactoryImpl();
      final actual = factory.fromType(CategoricalDataEncoderType.ordinal);
      expect(actual.runtimeType, OrdinalEncoder);
    });

    test('should create one-hot encoder', () {
      final factory = const CategoricalDataEncoderFactoryImpl();
      final actual = factory.oneHot();
      expect(actual.runtimeType, OneHotEncoder);
    });

    test('should create ordinal encoder', () {
      final factory = const CategoricalDataEncoderFactoryImpl();
      final actual = factory.ordinal();
      expect(actual.runtimeType, OrdinalEncoder);
    });
  });
}
