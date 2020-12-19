import 'dart:convert';

import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';

KnnClassifier createKnnClassifierFromJson(String json) {
  if (json.isEmpty) {
    throw Exception('Provided JSON object is empty, please provide a proper '
        'JSON object');
  }

  final decodedJson = jsonDecode(json) as Map<String, dynamic>;

  return KnnClassifierImpl.fromJson(decodedJson);
}
