import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/validate_train_data.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_max_depth.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_min_samples_count.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_min_error.dart';
import 'package:ml_algo/src/tree_trainer/_helpers/create_decision_tree_trainer.dart';
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
  validateTreeSolverMinError(minError);
  validateTreeSolversMinSamplesCount(minSamplesCount);
  validateTreeSolverMaxDepth(maxDepth);

  final trainer = createDecisionTreeTrainer(trainData, targetName, minError,
    minSamplesCount, maxDepth);
  final treeRootNode = trainer.train(trainData.toMatrix(dtype));

  return dependencies
      .get<DecisionTreeClassifierFactory>()
      .create(treeRootNode, targetName, dtype);
}
