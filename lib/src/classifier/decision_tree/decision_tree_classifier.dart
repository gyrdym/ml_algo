import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree/greedy_classifier_dependencies.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/split_selector.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

abstract class DecisionTreeClassifier implements Classifier, Assessable {
  factory DecisionTreeClassifier.greedy(
      DataSet data,
      double minError,
      int minSamplesCount,
      int maxDepth,
  ) {
    final dependencies = getGreedyDecisionTreeDependencies(minError,
        minSamplesCount, maxDepth);
    final solver = _createSolver(data, dependencies);
    return DecisionTreeClassifierImpl(solver);
  }

  factory DecisionTreeClassifier.id3() => null;

  static DecisionTreeSolver _createSolver(DataSet data,
      Injector dependencies) => DecisionTreeSolver(
      data.toMatrix(),
      data.columnRanges,
      data.outcomeRange,
      data.rangeToEncoded,
      dependencies.getDependency<LeafDetector>(),
      dependencies.getDependency<DecisionTreeLeafLabelFactory>(),
      dependencies.getDependency<SplitSelector>(),
  );
}
