import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/regressor/knn_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

KnnRegressor createKnnRegressor({
  DataFrame fittingData,
  String targetName,
  int k,
  KernelType kernel = KernelType.gaussian,
  Distance distance = Distance.euclidean,
  DType dtype = DType.float32,
}) => knnRegressorInjector
    .get<KnnRegressorFactory>()
    .create(
  fittingData,
  targetName,
  k,
  kernel,
  distance,
  dtype,
);
