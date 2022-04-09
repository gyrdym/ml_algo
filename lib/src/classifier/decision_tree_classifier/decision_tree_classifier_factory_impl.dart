import 'dart:convert';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_max_depth.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_min_error.dart';
import 'package:ml_algo/src/helpers/validate_tree_solver_min_samples_count.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

class DecisionTreeClassifierFactoryImpl
    implements DecisionTreeClassifierFactory {
  const DecisionTreeClassifierFactoryImpl(this._treeTrainerFactory);

  final TreeTrainerFactory _treeTrainerFactory;

  @override
  DecisionTreeClassifier create(
    DataFrame trainData,
    String targetName,
    DType dtype,
    num minError,
    int minSamplesCount,
    int maxDepth,
    TreeAssessorType assessorType,
  ) {
    validateTreeSolverMinError(minError);
    validateTreeSolversMinSamplesCount(minSamplesCount);
    validateTreeSolverMaxDepth(maxDepth);

    final trainer = _treeTrainerFactory.createByType(
      TreeTrainerType.decision,
      trainData,
      targetName,
      minError,
      minSamplesCount,
      maxDepth,
      assessorType,
      TreeLeafLabelFactoryType.majority,
      TreeSplitSelectorType.greedy,
      assessorType,
      TreeSplitterType.greedy,
    );
    final treeRootNode = trainer.train(trainData.toMatrix(dtype));

    return DecisionTreeClassifierImpl(
      minError,
      minSamplesCount,
      maxDepth,
      treeRootNode,
      targetName,
      assessorType,
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
