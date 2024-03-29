import 'package:ml_algo/src/retrieval/kd_tree/exceptions/invalid_query_point_length.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_split_strategy.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

import '../../helpers.dart';

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

    test(
        'should find the closest neighbours for [2.79, -9.15, 6.56, -18.59, 13.53], leafSize=3',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]);
      final result = kdTree.query(sample, 3).toList();

      expect(result[0].index, 12);
      expect(result[1].index, 4);
      expect(result[2].index, 18);
      expect(result, hasLength(3));
    });

    test(
        'should find the closest neighbours for [2.79, -9.15, 6.56, -18.59, 13.53], leafSize=1',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]);
      final result = kdTree.query(sample, 3).toList();

      expect(result[0].index, 12);
      expect(result[1].index, 4);
      expect(result[2].index, 18);
      expect(result, hasLength(3));
    });

    test(
        'should find the closest neighbours for Iterable [2.79, -9.15, 6.56, -18.59, 13.53], leafSize=1',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = [2.79, -9.15, 6.56, -18.59, 13.53];
      final result = kdTree.queryIterable(sample, 3).toList();

      expect(result[0].index, 12);
      expect(result[1].index, 4);
      expect(result[2].index, 18);
      expect(result, hasLength(3));
    });

    test(
        'should find the closest neighbours for [2.79, -9.15, 6.56, -18.59, 13.53], leafSize=3, splitStrategy=KDTreeSplitStrategy.largestVariance',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false),
          leafSize: 3, splitStrategy: KDTreeSplitStrategy.largestVariance);
      final sample = Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]);
      final result = kdTree.query(sample, 3).toList();

      expect(result[0].index, 12);
      expect(result[1].index, 4);
      expect(result[2].index, 18);
      expect(result, hasLength(3));
    });

    test(
        'should find the closest neighbours for [13.98, -8.21, 17.01, -5.14, 14.49], leafSize=3',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([13.98, -8.21, 17.01, -5.14, 14.49]);
      final result = kdTree.query(sample, 4).toList();

      expect(result[0].index, 4);
      expect(result[1].index, 12);
      expect(result[2].index, 3);
      expect(result[3].index, 18);
      expect(result, hasLength(4));
    });

    test(
        'should find the closest neighbours for [13.98, -8.21, 17.01, -5.14, 14.49], leafSize=1',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = Vector.fromList([13.98, -8.21, 17.01, -5.14, 14.49]);
      final result = kdTree.query(sample, 4).toList();

      expect(result[0].index, 4);
      expect(result[1].index, 12);
      expect(result[2].index, 3);
      expect(result[3].index, 18);
      expect(result, hasLength(4));
    }, skip: false);

    test(
        'should find the closest neighbours for [-9.88, -5.66, -16.15, 4.46, 2.34]',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([-9.88, -5.66, -16.15, 4.46, 2.34]);
      final result = kdTree.query(sample, 20).toList();

      expect(result[0].index, 19);
      expect(result[1].index, 11);
      expect(result[2].index, 6);
      expect(result[3].index, 9);
      expect(result[4].index, 18);
      expect(result[5].index, 2);
      expect(result[6].index, 14);
      expect(result[7].index, 10);
      expect(result[8].index, 15);
      expect(result[9].index, 5);
      expect(result[10].index, 7);
      expect(result[10].index, 7);
      expect(result[11].index, 13);
      expect(result[12].index, 3);
      expect(result[13].index, 12);
      expect(result[14].index, 17);
      expect(result[15].index, 1);
      expect(result[16].index, 0);
      expect(result[17].index, 16);
      expect(result[18].index, 4);
      expect(result[19].index, 8);
      expect(result, hasLength(20));
    });

    test('should find the closest neighbours for [1, 2, 3, 4, 5]', () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([1, 2, 3, 4, 5]);
      final result = kdTree.query(sample, 1).toList();

      expect(result[0].index, 18);
      expect(result, hasLength(1));
    });

    test('should find the closest neighbours for [100, -222, 444, 0, 1]', () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([100, -222, 444, 0, 1]);
      final result = kdTree.query(sample, 1).toList();

      expect(result[0].index, 4);
      expect(result, hasLength(1));
    });

    test(
        'should find the closest neighbours for [7.63, -10.70, 15.09, 10.25, -18.16] for concievable amount of iterations, leafSize=1',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = Vector.fromList([7.63, -10.70, 15.09, 10.25, -18.16]);

      kdTree.query(sample, 1).toList();

      expect((kdTree as KDTreeImpl).searchIterationCount, 4);
    });

    test(
        'should find the closest neighbours for [7.63, -10.70, 15.09, 10.25, -18.16] for concievable amount of iterations, leafSize=3',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([7.63, -10.70, 15.09, 10.25, -18.16]);

      kdTree.query(sample, 1).toList();

      expect((kdTree as KDTreeImpl).searchIterationCount, 6);
    });

    test(
        'should find the closest neighbours for [12, 23, 22, 11, -20], k=1, leafSize=1, cosine distance',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = Vector.fromList([12, 23, 22, 11, -20]);
      final result = kdTree.query(sample, 1, Distance.cosine).toList();

      expect(result, hasLength(1));
      expect(result[0].index, 17);
    });

    test(
        'should find the closest neighbours for [12, 23, 22, 11, -20], k=2, leafSize=1, cosine distance',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = Vector.fromList([12, 23, 22, 11, -20]);
      final result = kdTree.query(sample, 2, Distance.cosine).toList();

      expect(result, hasLength(2));
      expect(result[0].index, 17);
      expect(result[1].index, 8);
    });

    test(
        'should find the closest neighbours for [12, 23, 22, 11, -20], k=3, leafSize=1, cosine distance',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = Vector.fromList([12, 23, 22, 11, -20]);
      final result = kdTree.query(sample, 3, Distance.cosine).toList();

      expect(result, hasLength(3));
      expect(result[0].index, 17);
      expect(result[1].index, 8);
      expect(result[2].index, 0);
    });

    test(
        'should find the closest neighbours for [12, 23, 22, 11, -20], k=3, leafSize=3, cosine distance',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([12, 23, 22, 11, -20]);
      final result = kdTree.query(sample, 3, Distance.cosine).toList();

      expect(result, hasLength(3));
      expect(result[0].index, 17);
      expect(result[1].index, 8);
      expect(result[2].index, 0);
    });

    test(
        'should find the closest neighbours for [12, 23, 22, 11, -20], k=3, leafSize=3, cosine distance for conceivable amount of iterations',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false),
          splitStrategy: KDTreeSplitStrategy.largestVariance, leafSize: 1);
      final sample = Vector.fromList([12, 23, 22, 11, -20]);

      kdTree.query(sample, 3, Distance.cosine).toList();

      expect((kdTree as KDTreeImpl).searchIterationCount, 13);
    });

    test(
        'should find the closest neighbours for [2.79, -9.15, 6.56, -18.59, 13.53], leafSize=3,  manhatten distance',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]);
      final result = kdTree.query(sample, 3, Distance.manhattan).toList();

      expect(result[0].index, 12);
      expect(result[1].index, 4);
      expect(result[2].index, 13);
      expect(result, hasLength(3));
    });

    test(
        'should find the closest neighbours for [2.79, -9.15, 6.56, -18.59, 13.53], leafSize=1,  manhatten distance',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 1);
      final sample = Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]);
      final result = kdTree.query(sample, 3, Distance.manhattan).toList();

      expect(result[0].index, 12);
      expect(result[1].index, 4);
      expect(result[2].index, 13);
      expect(result, hasLength(3));
    });

    test('should throw an exception if the query point is of invalid length',
        () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      final sample = Vector.fromList([1, 2, 3, 4, 5, 6]);

      expect(() => kdTree.query(sample, 1),
          throwsA(isA<InvalidQueryPointLength>()));
    });

    test('should persist points', () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);

      expect(
          kdTree.points,
          iterable2dAlmostEqualTo([
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
          ]));
    });

    test('should persist dtype', () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      expect(kdTree.dtype, DType.float32);
    });

    test('should persist leaf size', () {
      final kdTree = KDTree(DataFrame(data, headerExists: false), leafSize: 3);
      expect(kdTree.leafSize, 3);
    });

    test('should create from iterable', () {
      final kdTree =
          KDTree.fromIterable(data, dtype: DType.float64, leafSize: 13);

      expect(
          kdTree.points,
          iterable2dAlmostEqualTo([
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
          ]));
      expect(kdTree.dtype, DType.float64);
      expect(kdTree.leafSize, 13);
    });
  });
}
