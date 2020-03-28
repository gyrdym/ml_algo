import 'package:ml_algo/src/tree_trainer/_helpers/create_decision_tree_trainer.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_tech/unit_testing/readers/json.dart';
import 'package:test/test.dart';

void main() {
  group('TreeSolver', () {
    group('DecisionTreeSolver', () {
      final dataFrame = DataFrame.fromSeries([
        Series('col_1', <int>[10, 90, 23, 55]),
        Series('col_2', <int>[20, 51, 40, 10]),
        Series('col_3', <int>[1, 0, 0, 1], isDiscrete: true),
        Series('col_4', <int>[0, 0, 1, 0], isDiscrete: true),
        Series('col_5', <int>[0, 1, 0, 0], isDiscrete: true),
        Series('col_6', <int>[30, 34, 90, 22]),
        Series('col_7', <int>[40, 31, 50, 80]),
        Series('col_8', <int>[0, 0, 1, 2], isDiscrete: true),
      ]);

      test('should build a decision tree structure', () async {
        final snapshotFileName = 'test/tree_trainer/'
            'tree_solver_integration_test.json';
        final solver = createDecisionTreeTrainer(dataFrame, 'col_8', 0.3, 1, 3,
            DType.float32);
        final actual = solver.root.serialize();
        final expected = await readJSON(snapshotFileName);

        expect(actual, equals(expected));
      });
    });
  });
}
