import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// Returns bin IDs from binary representation of decimal numbers
///
/// Bin IDs will be used as keys of a hash map, a value of the hash map is a bin itself.
Vector getBinIdsFromBinaryRepresentation(Matrix binaryRepresentation) {
  final dimension = binaryRepresentation.columnsNum;
  final dtype = binaryRepresentation.dtype;
  final powersOfTwo = Vector.fromList(
      List.generate(dimension, (index) => 1 << index),
      dtype: dtype);

  return (binaryRepresentation * powersOfTwo).toVector();
}
