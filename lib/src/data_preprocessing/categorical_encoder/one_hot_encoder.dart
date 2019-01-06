import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class OneHotEncoder implements CategoricalDataEncoder {
  @override
  final Map<String, List<Object>> categories;

  @override
  final EncodeUnknownValueStrategy encodeUnknownValueStrategy;

  OneHotEncoder(Map<String, List<Object>> categories,
      [this.encodeUnknownValueStrategy = EncodeUnknownValueStrategy.throwError]) :
    categories = Map<String, List<Object>>.unmodifiable(categories);

  @override
  List<double> encode(String categoryLabel, Object value) {
    if (!categories.containsKey(categoryLabel)) {
      throw UnsupportedError('One hot encoding: unsupported category `$categoryLabel`');
    }

    final values = categories[categoryLabel];

    if (!values.contains(value)) {
      if (encodeUnknownValueStrategy == EncodeUnknownValueStrategy.throwError) {
        throw UnsupportedError('One hot encoding: unsupported value `$value` for the category `$categoryLabel`');
      } else {
        return List<double>.filled(values.length, 0.0);
      }
    }

    final targetIdx = values.indexOf(value);
    return List<double>.generate(values.length, (int idx) => idx == targetIdx ? 1.0 : 0.0);
  }
}
