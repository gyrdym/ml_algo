import 'package:ml_algo/src/knn_kernel/cosine_kernel.dart';
import 'package:ml_algo/src/knn_kernel/epanechnikov_kernel.dart';
import 'package:ml_algo/src/knn_kernel/gaussian_kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_json_converter.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type_json_keys.dart';
import 'package:ml_algo/src/knn_kernel/uniform_kernel.dart';
import 'package:test/test.dart';

void main() {
  group('KernelJsonConverter', () {
    const converter = KernelJsonConverter();
    const cosineKernel = CosineKernel();
    const epanechnikovKernel = EpanechnikovKernel();
    const gaussianKernel = GaussianKernel();
    const uniformKernel = UniformKernel();

    test('should encode cosine kernel', () {
      expect(converter.toJson(cosineKernel), cosineKernelEncodedValue);
    });

    test('should decode cosine kernel', () {
      expect(converter.fromJson(cosineKernelEncodedValue), cosineKernel);
    });

    test('should encode Epanechnikov kernel', () {
      expect(converter.toJson(epanechnikovKernel), epanechnikovKernelEncodedValue);
    });

    test('should decode Epanechnikov kernel', () {
      expect(converter.fromJson(epanechnikovKernelEncodedValue), epanechnikovKernel);
    });

    test('should encode Gaussian kernel', () {
      expect(converter.toJson(gaussianKernel), gaussianKernelEncodedValue);
    });

    test('should decode Gaussian kernel', () {
      expect(converter.fromJson(gaussianKernelEncodedValue), gaussianKernel);
    });

    test('should encode uniform kernel', () {
      expect(converter.toJson(uniformKernel), uniformKernelEncodedValue);
    });

    test('should decode uniform kernel', () {
      expect(converter.fromJson(uniformKernelEncodedValue), uniformKernel);
    });
  });
}
