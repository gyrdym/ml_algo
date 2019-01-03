import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class OneHotEncoder implements CategoricalDataEncoder {
  @override
  final Map<String, List<Object>> categories;

  @override
  final EncodeUnknownValueStrategy encodeUnknownValueStrategy;

  final Map<String, Map<Object, int>> _categoriesIndexed;

  OneHotEncoder(Map<String, List<Object>> dataDescr,
      [this.encodeUnknownValueStrategy = EncodeUnknownValueStrategy.error]) :
    categories = Map<String, List<Object>>.unmodifiable(dataDescr),
    _categoriesIndexed = Map.unmodifiable(
      Map<String, Map<Object, int>>.fromIterable(dataDescr.keys,
        key: (dynamic key) => key as String,
        value: (dynamic key) => Map<Object, int>.fromIterable(dataDescr[key],
            key: (dynamic value) => value,
            value: (dynamic value) => dataDescr[key].indexOf(value),
        ),
      ),
    );

  @override
  List<double> encode(String categoryLabel, Object value) {
    if (!_categoriesIndexed.containsKey(categoryLabel)) {
      throw UnsupportedError('One hot encoding: unsupported category `$categoryLabel`');
    }

    final values = _categoriesIndexed[categoryLabel];

    if (!values.containsKey(value)) {
      if (encodeUnknownValueStrategy == EncodeUnknownValueStrategy.error) {
        throw UnsupportedError('One hot encoding: unsupported value `$value` for the category `$categoryLabel`');
      } else {
        return List<double>.filled(values.length, 0.0);
      }
    }

    final targetIdx = values[value];
    return List<double>.generate(values.length, (int idx) => idx == targetIdx ? 1.0 : 0.0);
  }
}
