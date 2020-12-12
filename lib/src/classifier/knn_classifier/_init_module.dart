import 'package:ml_algo/src/classifier/knn_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/injector.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory_impl.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory_impl.dart';

void initKnnClassifierModule() {
  initCommonModule();

  knnClassifierInjector
    ..registerSingletonIf<KernelFactory>(
            () => const KernelFactoryImpl())

    ..registerSingletonIf<KnnSolverFactory>(
            () => const KnnSolverFactoryImpl())

    ..registerSingletonIf<KnnClassifierFactory>(
            () => const KnnClassifierFactoryImpl())

    ..registerSingletonIf<KnnRegressorFactory>(
            () => KnnRegressorFactoryImpl(
          knnClassifierInjector.get<KernelFactory>(),
          knnClassifierInjector.get<KnnSolverFactory>(),
        ));
}
