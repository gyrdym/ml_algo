import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:test/test.dart';

void main() {
  group('KnnClassifier', () {
    test('should deserialize v1 schema version', () async {
      final file = File('e2e/knn_classifier/knn_classifier_v1.json');
      final encodedData = await file.readAsString();
      final classifier = KnnClassifier.fromJson(encodedData);

      expect(classifier.distanceType, Distance.euclidean);
      expect(classifier.kernelType, KernelType.gaussian);
      expect(classifier.k, 5);
      expect(classifier.negativeLabel, isNaN);
      expect(classifier.positiveLabel, isNaN);
      expect(classifier.dtype, DType.float32);
      expect(classifier.targetNames, ['Species']);
    });
  });
}
