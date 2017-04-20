import 'package:dart_ml/dart_ml.dart' show VectorInterface, TypedVector, DataCategory, DataTrainTestSplitter;
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  TypedVector vector = new TypedVector.from([1.5, 2.0, 3.0, 5.5, 23.0, 45.0, 60.0, 78.0, 99.0]);

  List<TypedVector> matrix = [
    new TypedVector.from([1.5, 92.0, 34.0, 5.6, 23.0]),
    new TypedVector.from([11.5, 29.0, 32.0, 5.6, 23.0]),
    new TypedVector.from([1.55, 2.0, 3.0, 5.6, 23.0]),
    new TypedVector.from([21.5, 2.0, 31.0, 5.6, 23.0]),
    new TypedVector.from([145.15, 12.0, 13.0, 5.6, 23.0]),
    new TypedVector.from([10.5, 234.0, 3.0, 5.6, 23.0]),
    new TypedVector.from([1.05, 278.0, 3.0, 5.6, 23.0]),
    new TypedVector.from([21.5, 92.0, 35.0, 5.6, 23.0]),
    new TypedVector.from([15.5, 2.06, 3.0, 5.6, 23.0]),
    new TypedVector.from([13.5, 23.0, 3.0, 5.6, 23.0]),
    new TypedVector.from([1.5, 122.0, 35.0, 5.6, 23.0]),
    new TypedVector.from([133.5, 425.0, 53.0, 5.6, 23.0]),
    new TypedVector.from([144.5, 25.0, 3.50, 5.6, 23.0]),
    new TypedVector.from([166.5, 26.0, 83.0, 5.6, 23.0]),
    new TypedVector.from([61.5, 92.0, 3.0, 5.6, 23.0]),
    new TypedVector.from([19.5, 2.0, 3.0, 5.6, 23.0])
  ];

  group('Split a data to train/test samples', () {
    test('Vector split', () {
      Map<DataCategory, VectorInterface> result = DataTrainTestSplitter.splitVector(vector, 0.6);

      expect(result[DataCategory.TRAIN], equals([1.5, 2.0, 3.0, 5.5, 23.0]));
      expect(result[DataCategory.TEST], equals([45.0, 60.0, 78.0, 99.0]));
    });

    test('Matrix split', () {
      Map<DataCategory, List<VectorInterface>> result = DataTrainTestSplitter.splitMatrix(matrix, 0.7);

      expect(result[DataCategory.TRAIN].length, equals(11));
      expect(result[DataCategory.TEST].length, equals(5));
    });
  });
}