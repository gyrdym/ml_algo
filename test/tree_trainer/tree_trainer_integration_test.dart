import 'package:ml_algo/src/classifier/decision_tree_classifier/_init_module.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_injector.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

import '../fake_data_set.dart';
import '../majority_tree_data_mock.dart';

void main() {
  group('TreeTrainer', () {
    final targetName = 'col_8';
    final minErrorOnNode = 0.3;
    final minSamplesCountOnNode = 1;
    final maxDepth = 3;

    group('DecisionTreeTrainer', () {
      setUp(initDecisionTreeModule);

      tearDown(() {
        injector.clearAll();
        decisionTreeInjector.clearAll();
      });

      test('should build a decision tree', () {
        final trainer =
            decisionTreeInjector.get<TreeTrainerFactory>().createByType(
                  TreeTrainerType.decision,
                  fakeDataSet,
                  targetName,
                  minErrorOnNode,
                  minSamplesCountOnNode,
                  maxDepth,
                  TreeSplitAssessorType.majority,
                  TreeLeafLabelFactoryType.majority,
                  TreeSplitSelectorType.greedy,
                  TreeSplitAssessorType.majority,
                  TreeSplitterType.greedy,
                );
        final rootNode = trainer.train(fakeDataSet.toMatrix(DType.float32));
        final actual = rootNode.toJson();

        expect(actual, equals(majorityTreeDataMock));
      });
    });
  });
}
