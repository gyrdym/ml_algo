import 'package:ml_algo/src/algorithms/knn/kernel.dart';
import 'package:test/test.dart';

void main() {
  group('Kernel', () {
    test('uniform should always return 1', () {
      expect(uniformKernel(0), 1);
      expect(uniformKernel(100000), 1);
    });

    test('epanechnikov should return proper value', () {
      expect(epanechnikovKernel(0), 0.75);
      expect(epanechnikovKernel(1), 0);
      expect(epanechnikovKernel(10), -74.25);
    });

    test('cosine should return proper value', () {
      expect(cosineKernel(0), closeTo(0.7853, 1e-4));
      expect(cosineKernel(1), closeTo(0.0000, 1e-4));
      expect(cosineKernel(20), closeTo(0.7853, 1e-4));
    });

    test('gaussian should return proper value', () {
      expect(gaussianKernel(0), closeTo(0.3989, 1e-4));
      expect(gaussianKernel(1), closeTo(0.2419, 1e-4));
      expect(gaussianKernel(3), closeTo(0.0044, 1e-4));
      expect(gaussianKernel(10), closeTo(0.0000, 1e-4));
    });
  });
}
