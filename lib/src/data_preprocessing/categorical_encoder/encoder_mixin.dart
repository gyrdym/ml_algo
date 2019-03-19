import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

mixin EncoderMixin implements CategoricalDataEncoder {
  Map<String, Vector> _sourceToEncoded;
  Map<Vector, String> _encodedToSource;

  @override
  Matrix encode(Iterable<String> values) {
    _sourceToEncoded ??= _createSourceToEncodedMap(values.toList(
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

  Map<String, Vector> _createSourceToEncodedMap(List<String> values) {
    final categoryLabels = Set<String>.from(values).toList(growable: false);
    return Map<String, Vector>.fromIterable(categoryLabels,
      key: (dynamic value) => value as String,
      value: (dynamic value) => encodeLabel(value as String, categoryLabels),
    );
  }

  Map<Vector, String> _invertSourceToEncoded() =>
      Map.fromEntries(_sourceToEncoded.entries.map(
              (entry) => MapEntry(entry.value, entry.key)));
}
