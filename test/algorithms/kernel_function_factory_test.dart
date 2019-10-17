import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:test/test.dart';

void main() {
  group('KernelFunctionFactoryImpl', () {
    final factory = const KernelFunctionFactoryImpl();

    test('should create proper instance for kernels', () {
      expect([
        factory.createByType(Kernel.uniform) is KernelFn,
        factory.createByType(Kernel.epanechnikov) is KernelFn,
        factory.createByType(Kernel.cosine) is KernelFn,
        factory.createByType(Kernel.gaussian) is KernelFn,
      ], equals(List<bool>.filled(4, true)));
    });
  });
}
