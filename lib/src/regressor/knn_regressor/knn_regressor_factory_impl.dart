import 'dart:convert';

import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

class KnnRegressorFactoryImpl implements KnnRegressorFactory {
  const KnnRegressorFactoryImpl(this._kernelFnFactory, this._solverFactory);

  final KernelFactory _kernelFnFactory;
  final KnnSolverFactory _solverFactory;

  @override
  KnnRegressor create(
    DataFrame fittingData,
    String targetName,
    int k,
    KernelType kernelType,
    Distance distance,
    DType dtype,
  ) {
    final splits = featuresTargetSplit(
      fittingData,
      targetNames: [targetName],
    ).toList();

    final trainFeatures = splits[0];
    final trainOutcomes = splits[1];

    final solver = _solverFactory.create(
      trainFeatures.toMatrix(dtype),
      trainOutcomes.toMatrix(dtype),
      k,
      distance,
      true,
    );

    final kernel = _kernelFnFactory.createByType(kernelType);
    final targetIndex = [...fittingData.header].indexOf(targetName);

    return KnnRegressorImpl(
      targetName,
      targetIndex,
      solver,
      kernel,
      dtype,
    );
  }

  @override
  KnnRegressor fromJson(String json) {
    if (json.isEmpty) {
      throw Exception('Provided JSON object is empty, please provide a proper '
          'JSON object');
    }

    final decodedJson = jsonDecode(json) as Map<String, dynamic>;

    return KnnRegressorImpl.fromJson(decodedJson);
  }
}
