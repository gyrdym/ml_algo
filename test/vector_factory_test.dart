import 'package:dart_ml/dart_ml.dart' show VectorInterface, TypedVector, RegularVector;
import 'package:dart_ml/src/math/vector/vector_factory.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  group('Create a vector via `filled` constructor. ', () {
    test('Create typed vector: ', () {
      TypedVector vector = VectorFactory.createFilled(TypedVector, 5, 2.0);
      expect(vector, equals([2.0, 2.0, 2.0, 2.0, 2.0]));
    });
  });
}