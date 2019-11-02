import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory_impl.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_solver/splitter/nominal_splitter/nominal_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_solver/splitter/numerical_splitter/numerical_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_factory_impl.dart';
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
import 'package:ml_algo/src/tree_solver/tree_solver_factory.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_factory_impl.dart';

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

      ..registerSingleton<LogisticRegressorFactory>(
              (_) => const LogisticRegressorFactoryImpl())
          
      ..registerSingleton<SoftmaxRegressorFactory>(
              (_) => const SoftmaxRegressorFactoryImpl())

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
          ))

      ..registerSingleton<SequenceElementsDistributionCalculatorFactory>(
              (_) => const SequenceElementsDistributionCalculatorFactoryImpl())

      ..registerSingleton<NominalTreeSplitterFactory>(
              (_) => const NominalTreeSplitterFactoryImpl())

      ..registerSingleton<NumericalTreeSplitterFactory>(
              (_) => const NumericalTreeSplitterFactoryImpl())

      ..registerSingleton<TreeSplitAssessorFactory>(
              (_) => const TreeSplitAssessorFactoryImpl())

      ..registerSingleton<TreeSplitterFactory>(
              (injector) => TreeSplitterFactoryImpl(
                injector.getDependency<TreeSplitAssessorFactory>(),
                injector.getDependency<NominalTreeSplitterFactory>(),
                injector.getDependency<NumericalTreeSplitterFactory>(),
              ))

      ..registerSingleton<TreeSplitSelectorFactory>(
              (injector) => TreeSplitSelectorFactoryImpl(
                injector.getDependency<TreeSplitAssessorFactory>(),
                injector.getDependency<TreeSplitterFactory>(),
              ))
      
      ..registerSingleton<TreeLeafDetectorFactory>(
              (injector) => TreeLeafDetectorFactoryImpl(
                injector.getDependency<TreeSplitAssessorFactory>(),
              ))

      ..registerSingleton<TreeLeafLabelFactoryFactory>(
              (injector) => TreeLeafLabelFactoryFactoryImpl(
                injector.getDependency<SequenceElementsDistributionCalculatorFactory>(),
              ))

      ..registerSingleton<TreeSolverFactory>(
              (injector) => TreeSolverFactoryImpl(
                  injector.getDependency<TreeLeafDetectorFactory>(),
                  injector.getDependency<TreeLeafLabelFactoryFactory>(),
                  injector.getDependency<TreeSplitSelectorFactory>(),
              ));
