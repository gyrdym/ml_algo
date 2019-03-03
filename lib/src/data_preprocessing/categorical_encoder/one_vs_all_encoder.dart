import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_linalg/matrix.dart';

class OneVsAllEncoder implements CategoricalDataEncoder {
  OneVsAllEncoder({
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
    throw UnsupportedError('It is impossible to encode a single value with'
        'one-vs-all encoder');
  }

  @override
  void setCategoryValues(List<Object> values) {
    _values ??= _valuesExtractor.extractCategoryValues(values);
  }

  @override
  Matrix encodeAll(Iterable<Object> values) {
    setCategoryValues(values.toList(growable: false));
    final sourceEncoded = _values
        .map<List<double>>((value) => _makeValueOneVsAll(values, value))
        .toList(growable: false);
    return Matrix.from(sourceEncoded).transpose();
  }

  List<double> _makeValueOneVsAll(Iterable<Object> values,
      Object targetValue) =>
    values.map<double>((value) => value == targetValue ? 1.0 : 0.0)
        .toList(growable: false);
}
