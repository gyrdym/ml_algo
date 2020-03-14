import 'dart:convert';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_serializable_field.dart';
import 'package:ml_algo/src/common/serializing_rule/dtype_serializing_rule.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_factory.dart';

DecisionTreeClassifier createDecisionTreeClassifierFromJson(String json) {
  final decoded = jsonDecode(json) as Map<String, dynamic>;

  final classNames = decoded[classNamesField] as List<String>;
  final dtype = dtypeSerializingRule.inverse[decoded[dtypeField]];
  final decodedSolverData = decoded[solverField] as Map<String, dynamic>;

  final solverFactory = dependencies.getDependency<TreeSolverFactory>();
  final solver = solverFactory.createFromMap(decodedSolverData);

  return DecisionTreeClassifierImpl(solver, classNames.first, dtype);
}
