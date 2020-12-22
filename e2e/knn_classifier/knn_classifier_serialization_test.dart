import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('KnnClassifier', () {
    test('should deserialize v0 schema version', () async {
      final file = File('e2e/knn_classifier/knn_classifier_v0.json');
      final encodedData = await file.readAsString();
      final classifier = KnnClassifier.fromJson(encodedData);

      expect(classifier.distanceType, Distance.euclidean);
      expect(classifier.kernelType, KernelType.gaussian);
      expect(classifier.k, 5);
      expect(classifier.negativeLabel, isNull);
      expect(classifier.positiveLabel, isNull);
      expect(classifier.dtype, DType.float32);
      expect(classifier.targetNames, ['Species']);
    });
  });
}
