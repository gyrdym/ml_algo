import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/non_linear/decision_tree/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/solver_factory/greedy_solver.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

/// A factory that creates different presets of decision tree classifier
///
/// Decision tree is an algorithm that recursively splits the input data into
/// subsets until the bests possible data subsets will be found.
abstract class DecisionTreeClassifier implements Classifier, Assessable {
  /// Creates majority based decision tree classifier.
  ///
  /// Majority based decision tree - a simplest tree algorithm, that uses
  /// majority label on a tree node as a prediction.
  factory DecisionTreeClassifier.majority(
      DataFrame samples,
      {
        int targetId,
        String targetName,
        double minError,
        int minSamplesCount,
        int maxDepth,
      }
  ) {
    final solver = createGreedySolver(
      samples,
      targetId,
      targetName,
      minError,
      minSamplesCount,
      maxDepth,
    );
    return DecisionTreeClassifierImpl(solver);
  }
}
