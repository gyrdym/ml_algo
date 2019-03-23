import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

mixin EncoderMixin implements CategoricalDataEncoder {
  Map<String, Vector> _sourceToEncoded;
  Map<Vector, String> _encodedToSource;

  @override
  Matrix encode(Iterable<String> values) {
    if (values.isEmpty) {
      throw Exception('Empty category label list has been passed');
    }
    _sourceToEncoded = _createSourceToEncodedMap(values.toList(
        growable: false));
    return Matrix.rows(
        values.map((value) => _sourceToEncoded[value]).toList(growable: false),
        dtype: dtype);
  }

  @override
  Iterable<String> decode(Matrix encoded) {
    _encodedToSource = _invertSourceToEncoded();
    return List<String>.generate(encoded.rowsNum,
            (i) => _decodeSingle(encoded.getRow(i)));
  }

  String _decodeSingle(Vector encoded) =>
      _encodedToSource.entries.firstWhere(
          (entry) => entry.key == encoded,
          orElse: () => throw Exception('There is no source value for provided '
              'encoded value ($encoded)'),
      ).value;

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
