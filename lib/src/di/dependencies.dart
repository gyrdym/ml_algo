import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory_impl.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_factory_impl.dart';
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
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_dependency_tokens.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/logit/float64_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/float64_softmax_link_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory_impl.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory_impl.dart';

Injector get dependencies =>
    injector ??= Injector()
      ..registerSingleton<LinearOptimizerFactory>(
              () => const LinearOptimizerFactoryImpl())

      ..registerSingleton<RandomizerFactory>(
              () => const RandomizerFactoryImpl())

      ..registerSingleton<LearningRateGeneratorFactory>(
              () => const LearningRateGeneratorFactoryImpl())

      ..registerSingleton<InitialCoefficientsGeneratorFactory>(
              () => const InitialCoefficientsGeneratorFactoryImpl())

      ..registerDependency<ConvergenceDetectorFactory>(
              () => const ConvergenceDetectorFactoryImpl())

      ..registerSingleton<CostFunctionFactory>(
              () => const CostFunctionFactoryImpl())

      ..registerSingleton<LinkFunction>(
              () => const Float32InverseLogitLinkFunction(),
          dependencyName: float32InverseLogitLinkFunctionToken)

      ..registerSingleton<LinkFunction>(
              () => const Float64InverseLogitLinkFunction(),
          dependencyName: float64InverseLogitLinkFunctionToken)

      ..registerSingleton<LinkFunction>(
              () => const Float32SoftmaxLinkFunction(),
          dependencyName: float32SoftmaxLinkFunctionToken)

      ..registerSingleton<LinkFunction>(
              () => const Float64SoftmaxLinkFunction(),
          dependencyName: float64SoftmaxLinkFunctionToken)

      ..registerSingleton<SplitIndicesProviderFactory>(
              () => const SplitIndicesProviderFactoryImpl())

      ..registerSingleton<SoftmaxRegressorFactory>(
              () => const SoftmaxRegressorFactoryImpl())

      ..registerSingleton<KernelFactory>(
              () => const KernelFactoryImpl())

      ..registerDependency<KnnSolverFactory>(
              () => const KnnSolverFactoryImpl())

      ..registerSingleton<KnnClassifierFactory>(
              () => const KnnClassifierFactoryImpl())

      ..registerSingleton<KnnRegressorFactory>(
              () => KnnRegressorFactoryImpl(
                injector.get<KernelFactory>(),
                injector.get<KnnSolverFactory>(),
          ))

      ..registerSingleton<SequenceElementsDistributionCalculatorFactory>(
              () => const SequenceElementsDistributionCalculatorFactoryImpl())

      ..registerSingleton<NominalTreeSplitterFactory>(
              () => const NominalTreeSplitterFactoryImpl())

      ..registerSingleton<NumericalTreeSplitterFactory>(
              () => const NumericalTreeSplitterFactoryImpl())
      
      ..registerSingleton<TreeSplitAssessorFactory>(
              () => const TreeSplitAssessorFactoryImpl())

      ..registerSingleton<TreeSplitterFactory>(
              () => TreeSplitterFactoryImpl(
                injector.get<TreeSplitAssessorFactory>(),
                injector.get<NominalTreeSplitterFactory>(),
                injector.get<NumericalTreeSplitterFactory>(),
              ))

      ..registerSingleton<TreeSplitSelectorFactory>(
              () => TreeSplitSelectorFactoryImpl(
                injector.get<TreeSplitAssessorFactory>(),
                injector.get<TreeSplitterFactory>(),
              ))
      
      ..registerSingleton<TreeLeafDetectorFactory>(
              () => TreeLeafDetectorFactoryImpl(
                injector.get<TreeSplitAssessorFactory>(),
              ))

      ..registerSingleton<TreeLeafLabelFactoryFactory>(
              () => TreeLeafLabelFactoryFactoryImpl(
                injector.get<SequenceElementsDistributionCalculatorFactory>(),
              ))

      ..registerSingleton<TreeTrainerFactory>(
              () => TreeTrainerFactoryImpl(
                  injector.get<TreeLeafDetectorFactory>(),
                  injector.get<TreeLeafLabelFactoryFactory>(),
                  injector.get<TreeSplitSelectorFactory>(),
              ))

      ..registerSingleton<DecisionTreeClassifierFactory>(
              () => const DecisionTreeClassifierFactoryImpl());
