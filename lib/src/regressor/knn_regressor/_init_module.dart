import 'package:injector/injector.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/injector.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory_impl.dart';
import 'package:ml_algo/src/regressor/knn_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory_impl.dart';

Injector initKnnRegressorModule() {
  initCommonModule();

  return knnRegressorInjector
    ..registerSingletonIf<KernelFactory>(() => const KernelFactoryImpl())
    ..registerSingletonIf<KnnSolverFactory>(() => const KnnSolverFactoryImpl())
    ..registerSingletonIf<KnnRegressorFactory>(() => KnnRegressorFactoryImpl(
          knnRegressorInjector.get<KernelFactory>(),
          knnRegressorInjector.get<KnnSolverFactory>(),
        ));
}
