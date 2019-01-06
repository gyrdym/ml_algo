import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class OrdinalEncoder implements CategoricalDataEncoder {
  @override
  final Map<String, List<Object>> categories;

  @override
  final EncodeUnknownValueStrategy encodeUnknownValueStrategy;

  OrdinalEncoder(Map<String, Iterable<Object>> categories,
      [this.encodeUnknownValueStrategy = EncodeUnknownValueStrategy.throwError]) :
        categories = Map<String, List<Object>>.unmodifiable(categories);

  @override
  Iterable<double> encode(String categoryLabel, Object value) {
    if (!categories.containsKey(categoryLabel)) {
      throw UnsupportedError('One hot encoding: unsupported category `$categoryLabel`');
    }

    final values = categories[categoryLabel];

    if (!values.contains(value)) {
      if (encodeUnknownValueStrategy == EncodeUnknownValueStrategy.throwError) {
        throw UnsupportedError('Ordinal encoding: unsupported value `$value` for the category `$categoryLabel`');
      } else {
        return [0.0];
      }
    }

    final ordinalNum = values.indexOf(value).toDouble();
    return [ordinalNum + 1]; // plus one - to avoid zero value. Zero is reserved for unknown values
  }
}
