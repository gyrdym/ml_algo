import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory_impl.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:test/test.dart';

void main() {
  group('KernelFunctionFactoryImpl', () {
    final factory = const KernelFactoryImpl();

    test('should create proper instance for kernels', () {
      expect(factory.createByType(KernelType.uniform), isA<Kernel>());
      expect(factory.createByType(KernelType.epanechnikov), isA<Kernel>());
      expect(factory.createByType(KernelType.cosine), isA<Kernel>());
      expect(factory.createByType(KernelType.gaussian), isA<Kernel>());
      expect(() => factory.createByType(null), throwsUnsupportedError);
    });
  });
}
