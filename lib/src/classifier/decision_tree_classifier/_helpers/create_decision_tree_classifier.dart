import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/validate_train_data.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_max_depth.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_min_samples_count.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_minimal_error.dart';
import 'package:ml_algo/src/tree_solver/_helpers/create_decision_tree_solver.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

DecisionTreeClassifier createDecisionTreeClassifier(
  DataFrame trainData,
  String targetName,
  num minError,
  int minSamplesCount,
  int maxDepth,
  DType dtype,
) {
  validateTrainData(trainData, [targetName]);
  validateTreeSolverMinimalError(minError);
  validateTreeSolversMinSamplesCount(minSamplesCount);
  validateTreeSolverMaxDepth(maxDepth);

  final solver = createDecisionTreeSolver(
    trainData,
    targetName,
    minError,
    minSamplesCount,
    maxDepth,
    dtype,
  );

  return dependencies
      .getDependency<DecisionTreeClassifierFactory>()
      .create(solver, targetName, dtype);
}
