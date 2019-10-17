import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function.dart';
import 'package:test/test.dart';

void main() {
  group('Kernel', () {
    test('uniform should return 1/2 if |lambda| <= 1', () {
      expect(uniformKernel(0), 1/2);
      expect(uniformKernel(0.5), 1/2);
      expect(uniformKernel(1), 1/2);
    });

    test('uniform should return 0 if |lambda| > 1', () {
      expect(uniformKernel(1.01), 0);
      expect(uniformKernel(10), 0);
    });

    test('epanechnikov should return proper value if |lambda| <= 1', () {
      expect(epanechnikovKernel(0), 0.75);
      expect(epanechnikovKernel(1), 0);
    });

    test('epanechnikov should return proper value if |lambda| > 1', () {
      expect(epanechnikovKernel(1.01), 0);
      expect(epanechnikovKernel(10), 0);
    });

    test('cosine should return proper value if |lambda| <= 1', () {
      expect(cosineKernel(0), closeTo(0.7853, 1e-4));
      expect(cosineKernel(1), closeTo(0.0000, 1e-4));
    });

    test('cosine should return proper value if |lambda| > 1', () {
      expect(cosineKernel(1.01), 0);
      expect(cosineKernel(20), 0);
    });

    test('gaussian should return proper value', () {
      expect(gaussianKernel(0), closeTo(0.3989, 1e-4));
      expect(gaussianKernel(1), closeTo(0.2419, 1e-4));
      expect(gaussianKernel(3), closeTo(0.0044, 1e-4));
      expect(gaussianKernel(10), closeTo(0.0000, 1e-4));
    });
  });
}
