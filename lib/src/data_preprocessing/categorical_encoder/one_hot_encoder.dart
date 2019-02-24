import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_linalg/matrix.dart';

class OneHotEncoder implements CategoricalDataEncoder {
  OneHotEncoder({
    this.encodeUnknownValueStrategy = EncodeUnknownValueStrategy.throwError,
    CategoryValuesExtractor valuesExtractor =
    const CategoryValuesExtractorImpl<Object>(),
  }) : _valuesExtractor = valuesExtractor;

  @override
  final EncodeUnknownValueStrategy encodeUnknownValueStrategy;

  final CategoryValuesExtractor _valuesExtractor;

  List<Object> _values;

  @override
  List<double> encodeSingle(Object value) {
    if (!_values.contains(value)) {
      if (encodeUnknownValueStrategy == EncodeUnknownValueStrategy.throwError) {
        throw UnsupportedError('One hot encoding: unknown value `$value`');
      } else {
        return List<double>.filled(_values.length, 0.0);
      }
    }
    final targetIdx = _values.indexOf(value);
    return List<double>.generate(
        _values.length, (int idx) => idx == targetIdx ? 1.0 : 0.0);
  }

  @override
  void setCategoryValues(List<Object> values) {
    _values ??= _valuesExtractor.extractCategoryValues(values);
  }

  @override
  MLMatrix encodeAll(Iterable<Object> values) {
    setCategoryValues(values.toList(growable: false));
    return MLMatrix.from(values.map(encodeSingle).toList(growable: false));
  }
}
