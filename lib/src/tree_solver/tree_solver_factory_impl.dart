import 'package:ml_algo/src/tree_solver/decision_tree_solver.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_solver/tree_solver.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_factory.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_type.dart';
import 'package:ml_linalg/matrix.dart';

class TreeSolverFactoryImpl implements TreeSolverFactory {

  TreeSolverFactoryImpl(
      this._leafDetectorFactory,
      this._leafLabelFactoryFactory,
      this._splitSelectorFactory,
  );

  final TreeLeafDetectorFactory _leafDetectorFactory;
  final TreeLeafLabelFactoryFactory _leafLabelFactoryFactory;
  final TreeSplitSelectorFactory _splitSelectorFactory;

  @override
  TreeSolver createByType(
      TreeSolverType type,
      Matrix samples,
      Iterable<int> featureIndices,
      int targetIdx,
      Map<int, List<num>> featureIdxToUniqueValues,
      num minErrorOnNode,
      int minSamplesCount,
      int maxDepth,
      TreeSplitAssessorType assessorType,
      TreeLeafLabelFactoryType leafLabelFactoryType,
      TreeSplitSelectorType splitSelectorType,
      TreeSplitAssessorType splitAssessorType,
      TreeSplitterType splitterType,
  ) {
    final leafDetector = _leafDetectorFactory
        .create(assessorType, minErrorOnNode, minSamplesCount, maxDepth);

    final leafLabelFactory = _leafLabelFactoryFactory
        .createByType(leafLabelFactoryType);

    final splitSelector = _splitSelectorFactory
        .createByType(splitSelectorType, splitAssessorType, splitterType);

    switch (type) {
      case TreeSolverType.decision:
        return DecisionTreeSolver(
          samples,
          featureIndices,
          targetIdx,
          featureIdxToUniqueValues,
          leafDetector,
          leafLabelFactory,
          splitSelector,
        );

      default:
        throw UnsupportedError('Tree solver type $type is unsupported');
    }
  }
}
