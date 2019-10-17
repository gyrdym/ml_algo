import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function_factory.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

abstract class KnnClassifier implements Classifier {
  factory KnnClassifier(
      DataFrame fittingData,
      String targetName,
      int k,
      {
        Kernel kernel = Kernel.gaussian,
        Distance distance = Distance.euclidean,
        DType dtype = DType.float32,
      }
  ) {
    final splits = featuresTargetSplit(fittingData,
      targetNames: [targetName],
    ).toList();

    final kernelFnFactory = getDependencies()
        .getDependency<KernelFunctionFactory>();
    final kernelFn = kernelFnFactory.createByType(kernel);

    return KnnClassifierImpl(
      splits[0].toMatrix(dtype),
      splits[1].toMatrix(dtype),
      targetName,
      kernelFn,
    );
  }
}
