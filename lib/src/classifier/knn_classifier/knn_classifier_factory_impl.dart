import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function_factory.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

class KnnClassifierFactoryImpl implements KnnClassifierFactory {
  KnnClassifierFactoryImpl(this.kernelFnFactory, this.knnSolverFactory);

  final KernelFunctionFactory kernelFnFactory;
  final KnnSolverFactory knnSolverFactory;

  @override
  KnnClassifier create(
      DataFrame fittingData,
      String targetName,
      int k,
      Kernel kernel,
      Distance distance,
      DType dtype,
  ) {
    final splits = featuresTargetSplit(fittingData,
      targetNames: [targetName],
    ).toList();

    final kernelFn = kernelFnFactory.createByType(kernel);
    final solverFn = knnSolverFactory.create();

    return KnnClassifierImpl(
      splits[0].toMatrix(dtype),
      splits[1].toMatrix(dtype),
      targetName,
      kernelFn,
      k,
      distance,
      solverFn,
      dtype,
    );
  }
}
