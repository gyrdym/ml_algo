import 'package:ml_algo/src/_mixin/data_validation_mixin.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

class KnnClassifierFactoryImpl with DataValidationMixin implements
    KnnClassifierFactory {

  KnnClassifierFactoryImpl(this.kernelFnFactory, this.knnSolverFactory);

  final KernelFactory kernelFnFactory;
  final KnnSolverFactory knnSolverFactory;

  @override
  KnnClassifier create(
      DataFrame fittingData,
      String targetName,
      int k,
      KernelType kernel,
      Distance distance,
      DType dtype,
  ) {
    validateTrainData(fittingData, [targetName]);

    final splits = featuresTargetSplit(fittingData,
      targetNames: [targetName],
    ).toList();

    final featuresSplit = splits[0];
    final targetSplit = splits[1];

    final trainFeatures = featuresSplit.toMatrix(dtype);
    final trainLabels = targetSplit.toMatrix(dtype);
    final classLabels = targetSplit[targetName].isDiscrete
        ? targetSplit[targetName]
            .discreteValues
            .map((dynamic value) => value as num)
            .toList(growable: false)
        : targetSplit
            .toMatrix(dtype)
            .getColumn(0)
            .unique()
            .toList(growable: false);

    final kernelFn = kernelFnFactory.createByType(kernel);

    final solver = knnSolverFactory.create(
      trainFeatures,
      trainLabels,
      k,
      distance,
      true,
    );

    return KnnClassifierImpl(
      targetName,
      classLabels,
      kernelFn,
      solver,
      dtype,
    );
  }
}
