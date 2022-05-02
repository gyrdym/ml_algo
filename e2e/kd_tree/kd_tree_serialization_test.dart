import 'dart:convert';
import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() async {
  group('KDTree', () {
    test('should restore from a JSON file, dtype=DType.float32', () async {
      final file =
          await File('e2e/kd_tree/kd_tree_float_32_v1.json').readAsString();
      final json = jsonDecode(file) as Map<String, dynamic>;
      final tree = KDTree.fromJson(json);
      final neighbours = tree.query(Vector.fromList([6.5, 3.01, 4.5, 1.5]), 5);

      expect(tree.dtype, DType.float32);
      expect(tree.leafSize, 1);
      expect(neighbours, hasLength(5));
      expect(neighbours.toString(),
          '((Index: 75, Distance: 0.17349341930302867), (Index: 51, Distance: 0.21470911402365767), (Index: 65, Distance: 0.26095956499211426), (Index: 86, Distance: 0.29681616124778537), (Index: 56, Distance: 0.4172527193942372))');
    });

    test('should restore from a JSON file, dtype=DType.float64', () async {
      final file =
          await File('e2e/kd_tree/kd_tree_float_64_v1.json').readAsString();
      final json = jsonDecode(file) as Map<String, dynamic>;
      final tree = KDTree.fromJson(json);
      final neighbours = tree.query(
          Vector.fromList([6.5, 3.01, 4.5, 1.5], dtype: DType.float64), 5);

      expect(tree.dtype, DType.float64);
      expect(tree.leafSize, 1);
      expect(neighbours, hasLength(5));
      expect(neighbours.toString(),
          '((Index: 75, Distance: 0.17349351572897434), (Index: 51, Distance: 0.21470910553583905), (Index: 65, Distance: 0.2609597670139979), (Index: 86, Distance: 0.29681644159311693), (Index: 56, Distance: 0.41725292090050153))');
    });
  });
}
