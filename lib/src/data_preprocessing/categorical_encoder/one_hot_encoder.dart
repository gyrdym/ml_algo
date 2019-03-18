import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class OneHotEncoder implements CategoricalDataEncoder {
  OneHotEncoder({
    CategoryValuesExtractor valuesExtractor =
      const CategoryValuesExtractorImpl(),
  }) : _valuesExtractor = valuesExtractor;

  final CategoryValuesExtractor _valuesExtractor;

  Map<String, Vector> _sourceToEncoded;
  Map<Vector, String> _encodedToSource;

  @override
  Matrix encode(Iterable<String> values) {
    _sourceToEncoded ??= _createLabelToEncodedValueMap(values.toList(
        growable: false));
    return Matrix.rows(
        values.map((value) => _sourceToEncoded[value]).toList(growable: false));
  }

  @override
  Iterable<String> decode(Matrix encoded) {
    _encodedToSource ??= _invertSourceToEncoded();
    return List<String>.generate(encoded.rowsNum,
            (i) => _encodedToSource[encoded.getRow(i)]);
  }

  Map<String, Vector> _createLabelToEncodedValueMap(List<String> values) {
    final categoryLabels = _valuesExtractor.extractCategoryValues(values);
    return Map<String, Vector>.fromIterable(categoryLabels,
      key: (dynamic value) => value as String,
      value: (dynamic value) => _encodeLabel(value as String, categoryLabels),
    );
  }

  Vector _encodeLabel(String value, List<String> categoryLabels) {
    final valueIdx = categoryLabels.indexOf(value);
    final encodedCategorySource = List<double>.generate(
        categoryLabels.length,
            (int idx) => idx == valueIdx ? 1.0 : 0.0
    );
    return Vector.from(encodedCategorySource);
  }

  Map<Vector, String> _invertSourceToEncoded() =>
      Map.fromEntries(_sourceToEncoded.entries.map(
              (entry) => MapEntry(entry.value, entry.key)));
}
