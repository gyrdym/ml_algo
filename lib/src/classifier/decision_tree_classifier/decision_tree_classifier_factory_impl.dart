import 'dart:convert';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/helpers/validate_train_data.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_max_depth.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_min_error.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_min_samples_count.dart';
import 'package:ml_algo/src/tree_trainer/_helpers/create_decision_tree_trainer.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

class DecisionTreeClassifierFactoryImpl implements
    DecisionTreeClassifierFactory {

  const DecisionTreeClassifierFactoryImpl();

  @override
  DecisionTreeClassifier create(
      DataFrame trainData,
      num minError,
      int minSamplesCount,
      int maxDepth,
      String targetName,
      DType dtype,
  ) {
    validateTrainData(trainData, [targetName]);
    validateTreeSolverMinError(minError);
    validateTreeSolversMinSamplesCount(minSamplesCount);
    validateTreeSolverMaxDepth(maxDepth);

    final trainer = createDecisionTreeTrainer(
      trainData,
      targetName,
      minError,
      minSamplesCount,
      maxDepth,
    );
    final treeRootNode = trainer
        .train(trainData.toMatrix(dtype));

    return DecisionTreeClassifierImpl(
      minError,
      minSamplesCount,
      maxDepth,
      treeRootNode,
      targetName,
      dtype,
    );
  }

  @override
  DecisionTreeClassifier fromJson(String json) {
    if (json.isEmpty) {
      throw Exception('Provided JSON object is empty, please provide a proper '
          'JSON object');
    }

    final decodedJson = jsonDecode(json) as Map<String, dynamic>;

    return DecisionTreeClassifierImpl.fromJson(decodedJson);
  }
}
