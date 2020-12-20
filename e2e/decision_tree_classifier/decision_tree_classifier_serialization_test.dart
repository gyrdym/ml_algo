import 'dart:io';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('DecisionTreeClassifier', () {
    test('should deserialize v0 schema version', () async {
      final file = File('e2e/decision_tree_classifier/decision_tree_classifier_v0.json');
      final encodedData = await file.readAsString();
      final classifier = DecisionTreeClassifier.fromJson(encodedData);

      expect(classifier.dtype, DType.float32);
      expect(classifier.targetNames, ['Species']);
      expect(classifier.positiveLabel, isNull);
      expect(classifier.negativeLabel, isNull);
      expect(classifier.minSamplesCount, isNull);
      expect(classifier.maxDepth, isNull);
      expect(classifier.minError, isNull);
    });
  });
}
