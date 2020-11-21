import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/tree_trainer/decision_tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/tree_trainer/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

import '../fake_data_set.dart';
import '../majority_tree_data_mock.dart';

void main() {
  group('DecisionTreeTrainer', () {
    final minErrorOnNode = 0.3;
    final minSamplesCount = 1;
    final maxDepth = 3;
    final columnIndices = [0, 1, 2, 3, 4, 5, 6, 7];
    final targetIndex = 7;
    final featureToUniqueValues = {
      2: [1, 0],
      3: [0, 1],
      4: [0, 1],
    };

    group('with majority predictor', () {
      final treeSplitAssessor = const MajorityTreeSplitAssessor();
      final leafDetector = TreeLeafDetectorImpl(
        treeSplitAssessor,
        minErrorOnNode,
        minSamplesCount,
        maxDepth,
      );
      final distributionCalculator = const DistributionCalculatorImpl();
      final treeLeafLabelFactory = MajorityTreeLeafLabelFactory(
          distributionCalculator,
      );
      final numericalTreeSplitter = const NumericalTreeSplitterImpl();
      final nominalTreeSplitter = const NominalTreeSplitterImpl();
      final treeSplitter = GreedyTreeSplitter(
        treeSplitAssessor,
        numericalTreeSplitter,
        nominalTreeSplitter,
      );
      final treeSplitSelector = GreedyTreeSplitSelector(
        treeSplitAssessor,
        treeSplitter,
      );
      final trainer = DecisionTreeTrainer(
        columnIndices,
        targetIndex,
        featureToUniqueValues,
        leafDetector,
        treeLeafLabelFactory,
        treeSplitSelector,
      );

      test('should build a decision tree', () {
        final rootNode = trainer.train(fakeDataSet.toMatrix(DType.float32));
        final actual = rootNode.toJson();

        expect(actual, equals(majorityTreeDataMock));
      });
    });
  });
}
