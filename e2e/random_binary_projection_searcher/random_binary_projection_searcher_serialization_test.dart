import 'dart:convert';
import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() async {
  group('RandomBinaryProjectionSearcher', () {
    test('should restore from a JSON file, dtype=DType.float32', () async {
      final file = await File(
              'e2e/random_binary_projection_searcher/random_binary_projection_searcher_32_v1.json')
          .readAsString();
      final json = jsonDecode(file) as Map<String, dynamic>;
      final searcher = RandomBinaryProjectionSearcher.fromJson(json);
      final k = 5;
      final searchRadius = 3;
      final neighbours = searcher.query(
          Vector.fromList(
              [6.5, 3.01, 4.5, 1.5, 2.3, 22.3, 14.09, 2.9, 22.0, 11.22]),
          k,
          searchRadius);

      expect(searcher.columns, [
        'feature_1',
        'feature_2',
        'feature_3',
        'feature_4',
        'feature_5',
        'feature_6',
        'feature_7',
        'feature_8',
        'feature_9',
        'feature_10'
      ]);
      expect(searcher.digitCapacity, 6);
      expect(searcher.seed, 15);
      expect(neighbours, hasLength(5));
      expect(neighbours.toString(),
          '((Index: 160, Distance: 4646.84383479798), (Index: 562, Distance: 5358.518638579137), (Index: 648, Distance: 5403.194564329513), (Index: 591, Distance: 5522.45033929686), (Index: 938, Distance: 6083.799141983568))');
    });

    test('should restore from a JSON file, dtype=DType.float64', () async {
      final file = await File(
              'e2e/random_binary_projection_searcher/random_binary_projection_searcher_64_v1.json')
          .readAsString();
      final json = jsonDecode(file) as Map<String, dynamic>;
      final searcher = RandomBinaryProjectionSearcher.fromJson(json);
      final k = 5;
      final searchRadius = 3;
      final neighbours = searcher.query(
          Vector.fromList(
              [6.5, 3.01, 4.5, 1.5, 2.3, 22.3, 14.09, 2.9, 22.0, 11.22],
              dtype: DType.float64),
          k,
          searchRadius);

      expect(searcher.columns, [
        'feature_1',
        'feature_2',
        'feature_3',
        'feature_4',
        'feature_5',
        'feature_6',
        'feature_7',
        'feature_8',
        'feature_9',
        'feature_10'
      ]);
      expect(searcher.digitCapacity, 6);
      expect(searcher.seed, 15);
      expect(neighbours, hasLength(5));
      expect(neighbours.toString(),
          '((Index: 160, Distance: 4646.843905472263), (Index: 562, Distance: 5358.51853463377), (Index: 648, Distance: 5403.194350179125), (Index: 591, Distance: 5522.45040389316), (Index: 938, Distance: 6083.79924961192))');
    });
  });
}
