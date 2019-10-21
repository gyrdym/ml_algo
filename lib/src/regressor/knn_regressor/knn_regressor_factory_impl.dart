import 'package:ml_algo/src/_mixin/data_validation_mixin.dart';
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

class KnnRegressorFactoryImpl with DataValidationMixin
    implements KnnRegressorFactory {

  KnnRegressorFactoryImpl(this._kernelFnFactory, this._solverFactory);

  final KernelFactory _kernelFnFactory;
  final KnnSolverFactory _solverFactory;

  @override
  KnnRegressor create(
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

    final trainFeatures = splits[0];
    final trainOutcomes = splits[1];

    final solver = _solverFactory.create(
      trainFeatures.toMatrix(dtype),
      trainOutcomes.toMatrix(dtype),
      k,
      distance,
      true,
    );

    final kernelFn = _kernelFnFactory.createByType(kernel);

    return KnnRegressorImpl(
      targetName,
      solver,
      kernelFn,
      dtype,
    );
  }
}
