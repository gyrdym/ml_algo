import 'package:get_it/get_it.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/get_it.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory_impl.dart';
import 'package:ml_algo/src/regressor/knn_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory_impl.dart';

GetIt initKnnRegressorModule() {
  initCommonModule();

  return knnRegressorModule
    ..registerSingletonIf<KernelFactory>(const KernelFactoryImpl())

    ..registerSingletonIf<KnnSolverFactory>(const KnnSolverFactoryImpl())

    ..registerSingletonIf<KnnRegressorFactory>(
        KnnRegressorFactoryImpl(
          knnRegressorModule.get<KernelFactory>(),
          knnRegressorModule.get<KnnSolverFactory>(),
        ));
}
