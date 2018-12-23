class OneHotEncoder {
  final Map<String, Map<Object, int>> _categories;

  OneHotEncoder(Map<String, List<Object>> data) : _categories = Map.unmodifiable(
    Map<String, Map<Object, int>>.fromIterable(data.keys,
        key: (dynamic key) => key as String,
        value: (dynamic key) => Map<Object, int>.fromIterable(data[key],
            key: (dynamic value) => value,
            value: (dynamic value) => data[key].indexOf(value),
        ),
    ),
  );

  List<double> encode(String categoryLabel, Object value) {
    if (!_categories.containsKey(categoryLabel)) {
      throw UnsupportedError('One hot encoding: unsupported category `$categoryLabel`');
    }

    final values = _categories[categoryLabel];

    if (!values.containsKey(value)) {
      throw UnsupportedError('One hot encoding: unsupported value `$value` for the category `$categoryLabel`');
    }

    final targetIdx = values[value];
    return List<double>.generate(values.length, (int idx) => idx == targetIdx ? 1.0 : 0.0);
  }
}
