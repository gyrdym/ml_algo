import 'package:ml_algo/src/knn_kernel/cosine_kernel.dart';
import 'package:ml_algo/src/knn_kernel/epanechnikov_kernel.dart';
import 'package:ml_algo/src/knn_kernel/gaussian_kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_kernel/uniform_kernel.dart';
import 'package:test/test.dart';

void main() {
  group('Kernel', () {
    test('uniform should return 1/2 if |lambda| <= 1', () {
      const kernel = UniformKernel();

      expect(kernel.getWeightByDistance(0), 1 / 2);
      expect(kernel.getWeightByDistance(0.5), 1 / 2);
      expect(kernel.getWeightByDistance(1), 1 / 2);
    });

    test('uniform should return 0 if |lambda| > 1', () {
      const kernel = UniformKernel();

      expect(kernel.getWeightByDistance(1.01), 0);
      expect(kernel.getWeightByDistance(10), 0);
    });

    test('uniform should contain a proper type field', () {
      expect(const UniformKernel().type, KernelType.uniform);
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

    test('epanechnikov should contain a proper type field', () {
      expect(const EpanechnikovKernel().type, KernelType.epanechnikov);
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

    test('cosine should contain a proper type field', () {
      expect(const CosineKernel().type, KernelType.cosine);
    });

    test('gaussian should return proper value', () {
      const kernel = GaussianKernel();

      expect(kernel.getWeightByDistance(0), closeTo(0.3989, 1e-4));
      expect(kernel.getWeightByDistance(1), closeTo(0.2419, 1e-4));
      expect(kernel.getWeightByDistance(3), closeTo(0.0044, 1e-4));
      expect(kernel.getWeightByDistance(10), closeTo(0.0000, 1e-4));
    });

    test('gaussian should contain a proper type field', () {
      expect(const GaussianKernel().type, KernelType.gaussian);
    });
  });
}
