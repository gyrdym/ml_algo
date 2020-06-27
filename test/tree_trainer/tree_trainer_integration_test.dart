import 'package:ml_algo/src/classifier/decision_tree_classifier/_init_module.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/tree_trainer/_helpers/create_decision_tree_trainer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

import '../fake_data_set.dart';
import '../majority_tree_data_mock.dart';

void main() {
  group('TreeTrainer', () {
    group('DecisionTreeTrainer', () {
      setUp(initDecisionTreeModule);

      tearDown(injector.clearAll);

      test('should build a decision tree', () {
        final targetName = 'col_8';
        final minErrorOnNode = 0.3;
        final minSamplesCountOnNode = 1;
        final maxDepth = 3;

        final trainer = createDecisionTreeTrainer(fakeDataSet,
            targetName, minErrorOnNode, minSamplesCountOnNode, maxDepth);
        final rootNode = trainer.train(fakeDataSet.toMatrix(DType.float32));

        final actual = rootNode.toJson();

        expect(actual, equals(majorityTreeDataMock));
      });
    });
  });
}
