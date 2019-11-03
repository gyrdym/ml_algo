import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_solver/tree_solver.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_type.dart';
import 'package:ml_linalg/matrix.dart';

abstract class TreeSolverFactory {
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
  );
}
