import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/knn_kernel/cosine_kernel.dart';
import 'package:ml_algo/src/knn_kernel/epanechnikov_kernel.dart';
import 'package:ml_algo/src/knn_kernel/gaussian_kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type_json_keys.dart';
import 'package:ml_algo/src/knn_kernel/uniform_kernel.dart';

class KernelJsonConverter implements JsonConverter<Kernel, String> {
  const KernelJsonConverter();

  @override
  Kernel fromJson(String json) {
    switch (json) {
      case cosineKernelEncodedValue:
        return const CosineKernel();

      case epanechnikovKernelEncodedValue:
        return const EpanechnikovKernel();

      case gaussianKernelEncodedValue:
        return const GaussianKernel();

      case uniformKernelEncodedValue:
        return const UniformKernel();

      default:
        throw UnsupportedError('Unsupported kernel json $json');
    }
  }

  @override
  String toJson(Kernel kernel) => kernel.toJson();
}
