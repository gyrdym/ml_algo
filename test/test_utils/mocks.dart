import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_selector/split_selector.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/splitter.dart' as decision_tree_splitter;
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
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
    implements InitialWeightsGeneratorFactory {}

class InitialWeightsGeneratorMock extends Mock
    implements InitialWeightsGenerator {}

class LinkFunctionMock extends Mock implements LinkFunction {}

class LinkFunctionFactoryMock extends Mock implements LinkFunctionFactory {}

class LinearOptimizerFactoryMock extends Mock
    implements LinearOptimizerFactory {}

class LinearOptimizerMock extends Mock implements LinearOptimizer {}

class ConvergenceDetectorFactoryMock extends Mock
    implements ConvergenceDetectorFactory {}

class ConvergenceDetectorMock extends Mock implements ConvergenceDetector {}

class SplitterMock extends Mock implements Splitter {}

class AssessableMock extends Mock implements Assessable {}

class SplitAssessorMock extends Mock implements SplitAssessor {}

class DecisionTreeSplitterMock extends Mock implements
    decision_tree_splitter.Splitter {}

class NumericalSplitterMock extends Mock implements NumericalSplitter {}

class NominalSplitterMock extends Mock implements NominalSplitter {}

class DistributionCalculatorMock extends Mock implements
    SequenceElementsDistributionCalculator {}

class LeafDetectorMock extends Mock implements LeafDetector {}

class LeafLabelFactoryMock extends Mock implements
    DecisionTreeLeafLabelFactory {}

class SplitSelectorMock extends Mock implements SplitSelector {}

class DecisionTreeNodeMock extends Mock implements DecisionTreeNode {}

class DecisionTreeSolverMock extends Mock implements DecisionTreeSolver {}

LearningRateGeneratorFactoryMock createLearningRateGeneratorFactoryMock({
  Map<LearningRateType, LearningRateGenerator> generators,
}) {
  final factory = LearningRateGeneratorFactoryMock();
  generators.forEach((LearningRateType type, LearningRateGenerator generator) {
    when(factory.fromType(type)).thenReturn(generator);
  });
  return factory;
}

RandomizerFactory createRandomizerFactoryMock({
  Map<int, RandomizerMock> randomizers,
}) {
  final factory = RandomizerFactoryMock();
  randomizers.forEach((int seed, Randomizer randomizer) {
    when(factory.create(seed)).thenReturn(randomizer);
  });
  return factory;
}

InitialWeightsGeneratorFactory createInitialWeightsGeneratorFactoryMock({
  Map<InitialWeightsType, InitialWeightsGenerator> generators,
}) {
  final factory = InitialWeightsGeneratorFactoryMock();
  generators
      .forEach((InitialWeightsType type, InitialWeightsGenerator generator) {
    when(factory.fromType(type, any)).thenReturn(generator);
  });
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

LinearOptimizerFactory createGradientOptimizerFactoryMock(
    LinearOptimizer optimizer) {

  final factory = LinearOptimizerFactoryMock();

  when(factory.createByType(
    argThat(isNotNull),
    argThat(isNotNull),
    argThat(isNotNull),
    dtype: anyNamed('dtype'),
    costFunction: anyNamed('costFunction'),
    learningRateType: anyNamed('learningRateType'),
    initialWeightsType: anyNamed('initialWeightsType'),
    initialLearningRate: anyNamed('initialLearningRate'),
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate'),
    iterationLimit: anyNamed('iterationLimit'),
    lambda: anyNamed('lambda'),
    batchSize: anyNamed('batchSize'),
    randomSeed: anyNamed('randomSeed'),
    isFittingDataNormalized: anyNamed('isFittingDataNormalized')
  )).thenReturn(optimizer);

  return factory;
}
