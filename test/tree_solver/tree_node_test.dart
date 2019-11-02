import 'package:ml_algo/src/tree_solver/tree_node.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_tech/unit_testing/readers/json.dart';
import 'package:test/test.dart';

void main() {
  group('TreeNode', () {
    test('should serialize itself', () async {
      final dummyTestFn = (Vector v) => true;

      final child31 = TreeNode(dummyTestFn, 600, null, [],
          null, 3);
      final child32 = TreeNode(dummyTestFn, 700, null, [],
          null, 3);
      final child33 = TreeNode(dummyTestFn, 800, null, [],
          null, 3);
      final child34 = TreeNode(dummyTestFn, 900, null, [],
          null, 3);
      final child35 = TreeNode(dummyTestFn, 900, null, [],
          null, 3);
      final child36 = TreeNode(dummyTestFn, 901, null, [],
          null, 3);
      final child37 = TreeNode(dummyTestFn, 911, null, [],
          null, 3);

      final child21 = TreeNode(dummyTestFn, 100, null, [
        child31,
        child32,
      ], null, 2);
      final child22 = TreeNode(dummyTestFn, 200, null, [
        child33,
        child34,
      ], null, 2);
      final child23 = TreeNode(dummyTestFn, 300, null, [
        child35,
        child36,
      ], null, 2);
      final child24 = TreeNode(dummyTestFn, 400, null, [
        child37,
      ], null, 2);

      final child25 = TreeNode(dummyTestFn, 500, null, [],
          null, 2);

      final child11 = TreeNode(dummyTestFn, 10, null, [
        child21,
        child22
      ], null, 1);
      final child12 = TreeNode(dummyTestFn, 12, null, [
        child23,
        child24,
      ], null, 1);
      final child13 = TreeNode(dummyTestFn, 13, null, [
        child25,
      ], null, 1);

      final root = TreeNode(dummyTestFn, null, null, [
        child11,
        child12,
        child13,
      ], null);

      final snapshotFileName = 'test/tree_solver/tree_node_test.json';
      final actual = root.serialize();
      final expected = await readJSON(snapshotFileName);

      expect(actual, expected);
    });
  });
}
