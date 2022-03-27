import 'package:ml_algo/src/retrieval/kd_tree/exceptions/invalid_query_point_length.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('KDTree', () {
    final data = [
      [3.43, 10.91, 11.62, -12.93, -11.66],
      [19.41, -4.96, 3.99, 16.35, 10.57],
      [11.30, 8.89, -17.66, -5.17, 16.20],
      [-8.13, -5.23, 18.01, 1.97, 9.08],
      [13.98, -8.21, 17.01, -5.14, 14.49],
      [-17.65, 13.10, 5.82, 8.61, 14.41],
      [4.16, -4.72, -3.71, -2.32, -13.70],
      [7.29, 11.16, -9.51, -1.89, -18.94],
      [19.81, 3.17, 14.27, 0.05, -17.93],
      [-9.63, 18.82, -14.40, -1.91, -6.58],
      [-10.95, -19.58, 9.05, 17.39, 3.30],
      [4.08, -13.19, -5.71, 18.56, -0.13],
      [2.79, -9.15, 6.56, -18.59, 13.53],
      [-7.56, 11.97, 6.55, -7.54, 15.90],
      [-15.97, -15.95, 7.71, 9.70, 16.94],
      [-15.01, 16.12, -10.42, -17.61, 6.27],
      [7.63, -10.70, 15.09, 10.25, -18.16],
      [0.05, 9.74, 7.08, 15.49, -17.99],
      [-6.48, 1.10, 9.28, 0.90, 6.09],
      [-9.88, -5.66, -16.15, 4.46, 2.34],
    ];

    final kdTree = KDTree(DataFrame(data, headerExists: false), leafSie: 3);

    test(
        'should find the closest neighbours for [2.79, -9.15, 6.56, -18.59, 13.53]',
        () {
      final sample = Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]);
      final result = kdTree.query(sample, 3).toList();

      expect(
          (kdTree as KDTreeImpl).searchIterationCount, lessThanOrEqualTo(15));
      expect(result[0].index, 12);
      expect(result[1].index, 4);
      expect(result[2].index, 18);
      expect(result, hasLength(3));
    });

    test(
        'should find the closest neighbours for [13.98, -8.21, 17.01, -5.14, 14.49]',
        () {
      final sample = Vector.fromList([13.98, -8.21, 17.01, -5.14, 14.49]);
      final result = kdTree.query(sample, 4).toList();

      expect(
          (kdTree as KDTreeImpl).searchIterationCount, lessThanOrEqualTo(15));
      expect(result[0].index, 4);
      expect(result[1].index, 12);
      expect(result[2].index, 3);
      expect(result[3].index, 18);
      expect(result, hasLength(4));
    });

    test(
        'should find the closest neighbours for [-9.88, -5.66, -16.15, 4.46, 2.34]',
        () {
      final sample = Vector.fromList([-9.88, -5.66, -16.15, 4.46, 2.34]);
      final result = kdTree.query(sample, 20).toList();

      expect(result[0].index, 19);
      expect(result[1].index, 11);
      expect(result[2].index, 6);
      expect(result[3].index, 9);
      expect(result[4].index, 18);
      expect(result, hasLength(20));
    });

    test('should find the closest neighbours for [1, 2, 3, 4, 5]', () {
      final sample = Vector.fromList([1, 2, 3, 4, 5]);
      final result = kdTree.query(sample, 1).toList();

      expect(result[0].index, 18);
      expect(result, hasLength(1));
    });

    test('should find the closest neighbours for [100, -222, 444, 0, 1]', () {
      final sample = Vector.fromList([100, -222, 444, 0, 1]);
      final result = kdTree.query(sample, 1).toList();

      expect(result[0].index, 4);
      expect(result, hasLength(1));
    });

    test('should throw an exception if the query point is of invalid length',
        () {
      final sample = Vector.fromList([1, 2, 3, 4, 5, 6]);

      expect(() => kdTree.query(sample, 1),
          throwsA(isA<InvalidQueryPointLength>()));
    });
  });
}
