import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory_impl.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_factory_impl.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_factory.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_factory_impl.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory_impl.dart';

Injector get dependencies =>
    injector ??= Injector()
      ..registerSingleton<LinearOptimizerFactory>(
              (_) => const LinearOptimizerFactoryImpl())

      ..registerSingleton<RandomizerFactory>(
              (_) => const RandomizerFactoryImpl())

      ..registerSingleton<LearningRateGeneratorFactory>(
              (_) => const LearningRateGeneratorFactoryImpl())

      ..registerSingleton<InitialCoefficientsGeneratorFactory>(
              (_) => const InitialCoefficientsGeneratorFactoryImpl())

      ..registerDependency<ConvergenceDetectorFactory>(
              (_) => const ConvergenceDetectorFactoryImpl())

      ..registerSingleton<CostFunctionFactory>(
              (_) => const CostFunctionFactoryImpl())

      ..registerSingleton<LinkFunctionFactory>(
              (_) => const LinkFunctionFactoryImpl())

      ..registerSingleton<DataSplitterFactory>(
              (_) => const DataSplitterFactoryImpl())

      ..registerSingleton<KernelFactory>(
              (_) => const KernelFactoryImpl())

      ..registerDependency<KnnSolverFactory>(
              (_) => const KnnSolverFactoryImpl())

      ..registerSingleton<KnnClassifierFactory>(
              (_) => const KnnClassifierFactoryImpl())

      ..registerSingleton<KnnRegressorFactory>(
              (injector) => KnnRegressorFactoryImpl(
                injector.getDependency<KernelFactory>(),
                injector.getDependency<KnnSolverFactory>(),
          ));
