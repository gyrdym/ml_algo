import 'dart:io';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('DecisionTreeClassifier', () {
    test('should deserialize v1 schema version', () async {
      final file = File('e2e/decision_tree_classifier/decision_tree_classifier_v1.json');
      final encodedData = await file.readAsString();
      final classifier = DecisionTreeClassifier.fromJson(encodedData);

      expect(classifier.dtype, DType.float64);
      expect(classifier.targetNames, ['Species']);
      expect(classifier.positiveLabel, isNaN);
      expect(classifier.negativeLabel, isNaN);
      expect(classifier.minSamplesCount, 5);
      expect(classifier.maxDepth, 4);
      expect(classifier.minError, 0.3);
    });
  });
}
