import 'package:injector/injector.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_factory_impl.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory_impl.dart';

Injector injector;

Injector getDependencies() =>
    injector ??= Injector()
          ..registerSingleton<LinearOptimizerFactory>(
                  (_) => const LinearOptimizerFactoryImpl())

          ..registerSingleton<RandomizerFactory>(
                  (_) => const RandomizerFactoryImpl())

          ..registerSingleton<LearningRateGeneratorFactory>(
                  (_) => const LearningRateGeneratorFactoryImpl())

          ..registerSingleton<InitialWeightsGeneratorFactory>(
                  (_) => const InitialWeightsGeneratorFactoryImpl())

          ..registerDependency<ConvergenceDetectorFactory>(
                  (_) => const ConvergenceDetectorFactoryImpl())

          ..registerSingleton<CostFunctionFactory>(
                  (_) => const CostFunctionFactoryImpl())

          ..registerSingleton<LinkFunctionFactory>(
                  (_) => const LinkFunctionFactoryImpl());
