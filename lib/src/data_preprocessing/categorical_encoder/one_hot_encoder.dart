import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class OneHotEncoder implements CategoricalDataEncoder {
  @override
  final Map<String, List<Object>> categories;

  final Map<String, Map<Object, int>> _categoriesIndexed;

  OneHotEncoder(Map<String, List<Object>> dataDescr) :
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
      throw UnsupportedError('One hot encoding: unsupported value `$value` for the category `$categoryLabel`');
    }

    final targetIdx = values[value];
    return List<double>.generate(values.length, (int idx) => idx == targetIdx ? 1.0 : 0.0);
  }
}
