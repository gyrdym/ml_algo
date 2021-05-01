import 'dart:convert';

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

class KnnClassifierFactoryImpl implements KnnClassifierFactory {
  const KnnClassifierFactoryImpl(
    this._kernelFactory,
    this._solverFactory,
  );

  final KernelFactory _kernelFactory;
  final KnnSolverFactory _solverFactory;

  @override
  KnnClassifier create(
    DataFrame trainData,
    String targetName,
    int k,
    KernelType kernelType,
    Distance distance,
    String columnPrefix,
    DType dtype,
  ) {
    final splits = featuresTargetSplit(
      trainData,
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
    final kernel = _kernelFactory.createByType(kernelType);
    final solver = _solverFactory.create(
      trainFeatures,
      trainLabels,
      k,
      distance,
      true,
    );

    return KnnClassifierImpl(
      targetName,
      classLabels,
      kernel,
      solver,
      columnPrefix,
      dtype,
    );
  }

  @override
  KnnClassifier fromJson(String json) {
    if (json.isEmpty) {
      throw Exception('Provided JSON object is empty, please provide a proper '
          'JSON object');
    }

    final decodedJson = jsonDecode(json) as Map<String, dynamic>;

    return KnnClassifierImpl.fromJson(decodedJson);
  }
}
