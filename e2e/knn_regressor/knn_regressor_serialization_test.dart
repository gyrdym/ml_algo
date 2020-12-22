import 'dart:io';

import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('KnnRegressor', () {
    test('should deserialize v0 schema version', () async {
      final file = File('e2e/knn_regressor/knn_regressor_v0.json');
      final encodedData = await file.readAsString();
      final regressor = KnnRegressor.fromJson(encodedData);

      expect(regressor.distanceType, Distance.euclidean);
      expect(regressor.kernelType, KernelType.gaussian);
      expect(regressor.k, 5);
      expect(regressor.dtype, DType.float32);
      expect(regressor.targetNames, ['col_13']);
    });
  });
}
