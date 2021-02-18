import 'package:get_it/get_it.dart';
import 'package:ml_algo/src/classifier/knn_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/get_it.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory_impl.dart';

GetIt initKnnClassifierModule() {
  initCommonModule();

  return knnClassifierModule
    ..registerSingletonIf<KernelFactory>(const KernelFactoryImpl())

    ..registerSingletonIf<KnnSolverFactory>(const KnnSolverFactoryImpl())

    ..registerSingletonIf<KnnClassifierFactory>(KnnClassifierFactoryImpl(
      knnClassifierModule.get<KernelFactory>(),
      knnClassifierModule.get<KnnSolverFactory>(),
    ));
}
