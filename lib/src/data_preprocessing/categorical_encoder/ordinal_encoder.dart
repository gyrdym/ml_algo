import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_linalg/matrix.dart';

class OrdinalEncoder implements CategoricalDataEncoder {
  OrdinalEncoder({
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
        throw UnsupportedError('Ordinal encoding: unsupported value `$value`');
      } else {
        return [0.0];
      }
    }
    final ordinalNum = _values.indexOf(value).toDouble();
    return [
      ordinalNum + 1
    ]; // plus one - to avoid zero value. Zero is reserved for unknown values
  }

  @override
  void setCategoryValues(List<Object> values) {
    _values ??= _valuesExtractor.extractCategoryValues(values);
  }

  @override
  Matrix encodeAll(Iterable<Object> values) {
    setCategoryValues(values.toList(growable: false));
    return Matrix.from(values.map(encodeSingle).toList(growable: false));
  }
}
