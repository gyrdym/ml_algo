import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class OneHotEncoder implements CategoricalDataEncoder {
  @override
  final EncodeUnknownValueStrategy encodeUnknownValueStrategy;

  final CategoryValuesExtractor _valuesExtractor;

  List<Object> _values;

  OneHotEncoder({
    this.encodeUnknownValueStrategy = EncodeUnknownValueStrategy.throwError,
    CategoryValuesExtractor valuesExtractor,
  }) : _valuesExtractor = valuesExtractor;

  @override
  List<double> encode(Object value) {
    if (!_values.contains(value)) {
      if (encodeUnknownValueStrategy == EncodeUnknownValueStrategy.throwError) {
        throw UnsupportedError('One hot encoding: unknown value `$value`');
      } else {
        return List<double>.filled(_values.length, 0.0);
      }
    }
    final targetIdx = _values.indexOf(value);
    return List<double>.generate(_values.length, (int idx) => idx == targetIdx ? 1.0 : 0.0);
  }

  @override
  void setCategoryValues(List<Object> values) {
    _values ??= _valuesExtractor.extractCategoryValues(values);
  }
}
