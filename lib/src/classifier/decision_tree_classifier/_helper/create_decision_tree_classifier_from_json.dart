import 'dart:convert';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';

DecisionTreeClassifier createDecisionTreeClassifierFromJson(String json) {
  if (json.isEmpty) {
    throw Exception('Provided JSON object is empty, please provide a proper '
        'JSON object');
  }

  final decodedJson = jsonDecode(json) as Map<String, dynamic>;

  return DecisionTreeClassifierImpl.fromJson(decodedJson);
}
