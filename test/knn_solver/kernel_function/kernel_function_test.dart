import 'package:ml_algo/src/knn_solver/kernel_function/cosine_kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/epanechnikov_kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/gaussian_kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/uniform_kernel.dart';
import 'package:test/test.dart';

void main() {
  group('Kernel', () {
    test('uniform should return 1/2 if |lambda| <= 1', () {
      const kernel = UniformKernel();

      expect(kernel.getWeightByDistance(0), 1/2);
      expect(kernel.getWeightByDistance(0.5), 1/2);
      expect(kernel.getWeightByDistance(1), 1/2);
    });

    test('uniform should return 0 if |lambda| > 1', () {
      const kernel = UniformKernel();

      expect(kernel.getWeightByDistance(1.01), 0);
      expect(kernel.getWeightByDistance(10), 0);
    });

    test('epanechnikov should return proper value if |lambda| <= 1', () {
      const kernel = EpanechnikovKernel();

      expect(kernel.getWeightByDistance(0), 0.75);
      expect(kernel.getWeightByDistance(1), 0);
    });

    test('epanechnikov should return proper value if |lambda| > 1', () {
      const kernel = EpanechnikovKernel();

      expect(kernel.getWeightByDistance(1.01), 0);
      expect(kernel.getWeightByDistance(10), 0);
    });

    test('cosine should return proper value if |lambda| <= 1', () {
      const kernel = CosineKernel();

      expect(kernel.getWeightByDistance(0), closeTo(0.7853, 1e-4));
      expect(kernel.getWeightByDistance(1), closeTo(0.0000, 1e-4));
    });

    test('cosine should return proper value if |lambda| > 1', () {
      const kernel = CosineKernel();

      expect(kernel.getWeightByDistance(1.01), 0);
      expect(kernel.getWeightByDistance(20), 0);
    });

    test('gaussian should return proper value', () {
      const kernel = GaussianKernel();

      expect(kernel.getWeightByDistance(0), closeTo(0.3989, 1e-4));
      expect(kernel.getWeightByDistance(1), closeTo(0.2419, 1e-4));
      expect(kernel.getWeightByDistance(3), closeTo(0.0044, 1e-4));
      expect(kernel.getWeightByDistance(10), closeTo(0.0000, 1e-4));
    });
  });
}
