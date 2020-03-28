import 'dart:convert';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_serializable_field.dart';
import 'package:ml_algo/src/common/serializing_rule/dtype_serializing_rule.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_deserialize.dart' as tree_node;

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
  final dtype = dtypeSerializingRule.inverse[decoded[dtypeField]];
  final treeRootNodeData = decoded[rootNodeField] as Map<String, dynamic>;
  final treeRootNode = tree_node.deserialize(treeRootNodeData);

  return DecisionTreeClassifierImpl(treeRootNode, classNames.first, dtype);
}
