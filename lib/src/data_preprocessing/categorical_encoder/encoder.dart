import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// A categorical data encoder. Contains names and values of the categories
/// that supposed to be encoded and provides method for data encoding
abstract class CategoricalDataEncoder {
  /// Encodes passed categorical values to a numerical representation
  Matrix encode(Iterable<String> values);

  /// Decodes passed categorical encoded data to a source string representation
  Iterable<String> decode(Matrix values);

  /// Encodes a single categorical label
  Vector encodeLabel(String label, Iterable<String> categoryLabels);
}
