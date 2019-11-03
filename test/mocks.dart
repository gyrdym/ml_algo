import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/tree_solver/decision_tree_solver.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_solver/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/tree_solver/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_solver/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_solver/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_solver/tree_node.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

class RandomizerFactoryMock extends Mock implements RandomizerFactory {}

class RandomizerMock extends Mock implements Randomizer {}

class CostFunctionMock extends Mock implements CostFunction {}

class CostFunctionFactoryMock extends Mock implements CostFunctionFactory {}

class LearningRateGeneratorFactoryMock extends Mock
    implements LearningRateGeneratorFactory {}

class LearningRateGeneratorMock extends Mock implements
    LearningRateGenerator {}

class InitialWeightsGeneratorFactoryMock extends Mock
    implements InitialCoefficientsGeneratorFactory {}

class InitialWeightsGeneratorMock extends Mock
    implements InitialCoefficientsGenerator {}

class LinkFunctionMock extends Mock implements LinkFunction {}

class LinkFunctionFactoryMock extends Mock implements LinkFunctionFactory {}

class LinearOptimizerFactoryMock extends Mock
    implements LinearOptimizerFactory {}

class LinearOptimizerMock extends Mock implements LinearOptimizer {}

class ConvergenceDetectorFactoryMock extends Mock
    implements ConvergenceDetectorFactory {}

class ConvergenceDetectorMock extends Mock implements ConvergenceDetector {}

class DataSplitterMock extends Mock implements DataSplitter {}

class DataSplitterFactoryMock extends Mock implements DataSplitterFactory {}

class AssessableMock extends Mock implements Assessable {}

class TreeSplitAssessorMock extends Mock implements TreeSplitAssessor {}

class TreeSplitAssessorFactoryMock extends Mock implements
    TreeSplitAssessorFactory {}

class TreeSplitterMock extends Mock implements TreeSplitter {}

class TreeSplitterFactoryMock extends Mock implements TreeSplitterFactory {}

class NumericalTreeSplitterMock extends Mock implements NumericalTreeSplitter {}

class NumericalTreeSplitterFactoryMock extends Mock implements
    NumericalTreeSplitterFactory {}

class NominalTreeSplitterMock extends Mock implements NominalTreeSplitter {}

class NominalTreeSplitterFactoryMock extends Mock implements
    NominalTreeSplitterFactory {}

class DistributionCalculatorMock extends Mock implements
    SequenceElementsDistributionCalculator {}

class DistributionCalculatorFactoryMock extends Mock implements
    SequenceElementsDistributionCalculatorFactory {}

class TreeLeafDetectorMock extends Mock implements TreeLeafDetector {}

class TreeLeafDetectorFactoryMock extends Mock implements
    TreeLeafDetectorFactory {}

class TreeLeafLabelFactoryMock extends Mock implements TreeLeafLabelFactory {}

class TreeLeafLabelFactoryFactoryMock extends Mock implements
    TreeLeafLabelFactoryFactory {}

class TreeSplitSelectorMock extends Mock implements TreeSplitSelector {}

class TreeSplitSelectorFactoryMock extends Mock implements
    TreeSplitSelectorFactory {}

class TreeNodeMock extends Mock implements TreeNode {}

class TreeSolverMock extends Mock implements DecisionTreeSolver {}

class KernelFunctionFactoryMock extends Mock implements KernelFactory {}

class KnnSolverFactoryMock extends Mock implements KnnSolverFactory {}

class KnnClassifierFactoryMock extends Mock implements KnnClassifierFactory {}

class KnnClassifierMock extends Mock implements KnnClassifier {}

class KnnRegressorFactoryMock extends Mock implements KnnRegressorFactory {}

class KnnRegressorMock extends Mock implements KnnRegressor {}

class KnnSolverMock extends Mock implements KnnSolver {}

class KernelMock extends Mock implements Kernel {}

class LogisticRegressorMock extends Mock implements LogisticRegressor {}

class LogisticRegressorFactoryMock extends Mock implements
    LogisticRegressorFactory {}

class SoftmaxRegressorMock extends Mock implements SoftmaxRegressor {}

class SoftmaxRegressorFactoryMock extends Mock implements
    SoftmaxRegressorFactory {}

LearningRateGeneratorFactoryMock createLearningRateGeneratorFactoryMock(
    LearningRateGenerator generator) {
  final factory = LearningRateGeneratorFactoryMock();
  when(factory.fromType(any)).thenReturn(generator);
  return factory;
}

RandomizerFactory createRandomizerFactoryMock(Randomizer randomizer) {
  final factory = RandomizerFactoryMock();
  when(factory.create(any)).thenReturn(randomizer);
  return factory;
}

InitialCoefficientsGeneratorFactory createInitialWeightsGeneratorFactoryMock(
    InitialCoefficientsGenerator generator) {
  final factory = InitialWeightsGeneratorFactoryMock();
  when(factory.fromType(any, any)).thenReturn(generator);
  return factory;
}

LinkFunctionFactory createLinkFunctionFactoryMock(
    LinkFunction linkFunctionMock) {
  final factory = LinkFunctionFactoryMock();

  when(factory.createByType(argThat(isNotNull), dtype: anyNamed('dtype')))
      .thenReturn(linkFunctionMock);

  return factory;
}

CostFunctionFactory createCostFunctionFactoryMock(
    CostFunction costFunctionMock) {
  final costFunctionFactory = CostFunctionFactoryMock();

  when(costFunctionFactory.createByType(argThat(isNotNull),
      linkFunction: anyNamed('linkFunction'))).thenReturn(costFunctionMock);

  return costFunctionFactory;
}

ConvergenceDetectorFactory createConvergenceDetectorFactoryMock(
    ConvergenceDetector detector) {
  final convergenceDetectorFactory = ConvergenceDetectorFactoryMock();
  when(convergenceDetectorFactory.create(any, any)).thenReturn(detector);
  return convergenceDetectorFactory;
}

LinearOptimizerFactory createLinearOptimizerFactoryMock(
    LinearOptimizer optimizer) {
  final factory = LinearOptimizerFactoryMock();

  when(factory.createByType(
    any,
    any,
    any,
    dtype: anyNamed('dtype'),
    costFunction: anyNamed('costFunction'),
    learningRateType: anyNamed('learningRateType'),
    initialCoefficientsType: anyNamed('initialCoefficientsType'),
    initialLearningRate: anyNamed('initialLearningRate'),
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate'),
    iterationLimit: anyNamed('iterationLimit'),
    lambda: anyNamed('lambda'),
    regularizationType: anyNamed('regularizationType'),
    batchSize: anyNamed('batchSize'),
    randomSeed: anyNamed('randomSeed'),
    isFittingDataNormalized: anyNamed('isFittingDataNormalized')
  )).thenReturn(optimizer);

  return factory;
}

DataSplitterFactory createDataSplitterFactoryMock(DataSplitter dataSplitter) {
  final factory = DataSplitterFactoryMock();
  when(factory.createByType(any,
      numberOfFolds: anyNamed('numberOfFolds'),
      p: anyNamed('p')),
  ).thenReturn(dataSplitter);
  return factory;
}

KernelFactory createKernelFactoryMock(Kernel kernel) {
  final factory = KernelFunctionFactoryMock();
  when(factory.createByType(any)).thenReturn(kernel);
  return factory;
}

KnnSolverFactory createKnnSolverFactoryMock(KnnSolver solver) {
  final factory = KnnSolverFactoryMock();
  when(factory.create(any, any, any, any, any)).thenReturn(solver);
  return factory;
}

KnnClassifierFactory createKnnClassifierFactoryMock(KnnClassifier classifier) {
  final factory = KnnClassifierFactoryMock();
  when(factory.create(any, any, any, any, any)).thenReturn(classifier);
  return factory;
}

KnnRegressorFactory createKnnRegressorFactoryMock(KnnRegressor regressor) {
  final factory = KnnRegressorFactoryMock();
  when(factory.create(any, any, any, any, any, any)).thenReturn(regressor);
  return factory;
}

LogisticRegressorFactory createLogisticRegressorFactoryMock(
    LogisticRegressor logisticRegressor) {
  final factory = LogisticRegressorFactoryMock();
  when(factory.create(any, any, any, any, any, any, any, any, any))
      .thenReturn(logisticRegressor);
  return factory;
}

SoftmaxRegressorFactory createSoftmaxRegressorFactoryMock(
    SoftmaxRegressor softmaxRegressor) {
  final factory = SoftmaxRegressorFactoryMock();
  when(factory.create(any, any, any, any, any, any, any, any))
      .thenReturn(softmaxRegressor);
  return factory;
}

TreeSplitAssessorFactory createTreeSplitAssessorFactoryMock(
    TreeSplitAssessor splitAssessor) {
  final factory = TreeSplitAssessorFactoryMock();
  when(factory.createByType(any)).thenReturn(splitAssessor);
  return factory;
}

TreeSplitterFactory createTreeSplitterFactoryMock(TreeSplitter splitter) {
  final factory = TreeSplitterFactoryMock();
  when(factory.createByType(any, any)).thenReturn(splitter);
  return factory;
}

NumericalTreeSplitterFactory createNumericalTreeSplitterFactoryMock(
  NumericalTreeSplitter splitter,
) {
  final factory = NumericalTreeSplitterFactoryMock();
  when(factory.create()).thenReturn(splitter);
  return factory;
}

NominalTreeSplitterFactory createNominalTreeSplitterFactoryMock(
  NominalTreeSplitter splitter,
) {
  final factory = NominalTreeSplitterFactoryMock();
  when(factory.create()).thenReturn(splitter);
  return factory;
}

TreeLeafDetectorFactory createTreeLeafDetectorFactoryMock(
    TreeLeafDetector leafDetector) {
  final factory = TreeLeafDetectorFactoryMock();
  when(factory.create(any, any, any, any)).thenReturn(leafDetector);
  return factory;
}

TreeLeafLabelFactoryFactory createTreeLeafLabelFactoryFactoryMock(
  TreeLeafLabelFactory leafLabelFactory,
) {
  final factory = TreeLeafLabelFactoryFactoryMock();
  when(factory.createByType(any)).thenReturn(leafLabelFactory);
  return factory;
}

TreeSplitSelectorFactory createTreeSplitSelectorFactoryMock(
    TreeSplitSelector splitSelector) {
  final factory = TreeSplitSelectorFactoryMock();
  when(factory.createByType(any, any, any)).thenReturn(splitSelector);
  return factory;
}

SequenceElementsDistributionCalculatorFactory createDistributionCalculatorFactoryMock(
  SequenceElementsDistributionCalculator calculator,
) {
  final factory = DistributionCalculatorFactoryMock();
  when(factory.create()).thenReturn(calculator);
  return factory;
}
