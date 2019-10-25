import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/helpers/validate_train_data.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

KnnClassifier createKnnClassifier(
    DataFrame fittingData,
    String targetName,
    int k,
    KernelType kernelType,
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

  final kernelFactory = dependencies.getDependency<KernelFactory>();
  final kernel = kernelFactory.createByType(kernelType);

  final solverFactory = dependencies.getDependency<KnnSolverFactory>();

  final solver = solverFactory.create(
    trainFeatures,
    trainLabels,
    k,
    distance,
    true,
  );

  final knnClassifierFactory = dependencies
      .getDependency<KnnClassifierFactory>();

  return knnClassifierFactory.create(
    targetName,
    classLabels,
    kernel,
    solver,
    dtype,
  );
}
