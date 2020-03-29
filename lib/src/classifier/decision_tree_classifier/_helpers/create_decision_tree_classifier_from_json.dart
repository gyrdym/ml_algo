import 'dart:convert';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_serializable_field.dart';
import 'package:ml_algo/src/common/serializable/serializer.dart';
import 'package:ml_algo/src/common/serializing_rule/dtype_serializing_rule.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

DecisionTreeClassifier createDecisionTreeClassifierFromJson(String json) {
  if (json.isEmpty) {
    throw Exception('Provided JSON object is empty, please provide a proper '
        'JSON object');
  }

  final decoded = jsonDecode(json) as Map<String, dynamic>;

  if (decoded[classNamesField] == null) {
    throw Exception('Provided JSON object does not contain a `$classNamesField` '
        'field');
  }

  if (decoded[dtypeField] == null) {
    throw Exception('Provided JSON object does not contain a `$dtypeField` '
        'field');
  }

  if (decoded[rootNodeField] == null) {
    throw Exception('Provided JSON object does not contain a `$rootNodeField` '
        'field');
  }

  final classNames = (decoded[classNamesField] as List<dynamic>)
      .map((dynamic className) => className.toString());
  final targetClassName = classNames.first;
  final dtype = dtypeSerializingRule.inverse[decoded[dtypeField]];
  final treeNodeSerializer = dependencies.getDependency<Serializer<TreeNode>>();
  final treeRootNodeData = decoded[rootNodeField] as Map<String, dynamic>;
  final treeRootNode = treeNodeSerializer.deserialize(treeRootNodeData);

  return dependencies
      .getDependency<DecisionTreeClassifierFactory>()
      .create(treeRootNode, targetClassName, dtype);
}
