import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class OrdinalEncoder implements CategoricalDataEncoder {
  @override
  final EncodeUnknownValueStrategy encodeUnknownValueStrategy;

  final CategoryValuesExtractor _valuesExtractor;

  List<Object> _values;

  OrdinalEncoder({
    this.encodeUnknownValueStrategy = EncodeUnknownValueStrategy.throwError,
    CategoryValuesExtractor valuesExtractor =
        const CategoryValuesExtractorImpl<Object>(),
  }) : _valuesExtractor = valuesExtractor;

  @override
  Iterable<double> encode(Object value) {
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
}
