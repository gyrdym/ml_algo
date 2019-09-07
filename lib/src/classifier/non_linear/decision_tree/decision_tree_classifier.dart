import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/non_linear/decision_tree/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/solver_factory/greedy_solver.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

abstract class DecisionTreeClassifier implements Classifier, Assessable {
  factory DecisionTreeClassifier(
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