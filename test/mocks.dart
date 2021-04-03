import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory.dart';
import 'package:ml_algo/src/tree_trainer/decision_tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

class MetricFactoryMock extends Mock implements MetricFactory {}

class MetricMock extends Mock implements Metric {}

class EncoderFactoryMock extends Mock {
  Encoder create(DataFrame data, Iterable<String> targetNames);
}

class FeatureTargetSplitterMock extends Mock {
  Iterable<DataFrame> split(DataFrame samples, {
    Iterable<String> targetNames,
    Iterable<int> targetIndices,
  });
}

class ClassLabelsNormalizerMock extends Mock {
  Matrix normalize(Matrix classLabels, num positiveLabel, num negativeLabel);
}

class EncoderMock extends Mock implements Encoder {}

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

class LinearOptimizerFactoryMock extends Mock
    implements LinearOptimizerFactory {}

class LinearOptimizerMock extends Mock implements LinearOptimizer {}

class ConvergenceDetectorFactoryMock extends Mock
    implements ConvergenceDetectorFactory {}

class ConvergenceDetectorMock extends Mock implements ConvergenceDetector {}

class DataSplitterMock extends Mock implements SplitIndicesProvider {}

class DataSplitterFactoryMock extends Mock implements SplitIndicesProviderFactory {}

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
    DistributionCalculator {}

class DistributionCalculatorFactoryMock extends Mock implements
    DistributionCalculatorFactory {}

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

class TreeSolverMock extends Mock implements DecisionTreeTrainer {}

class KernelFunctionFactoryMock extends Mock implements KernelFactory {}

class KnnSolverFactoryMock extends Mock implements KnnSolverFactory {}

class KnnClassifierFactoryMock extends Mock implements KnnClassifierFactory {}

class KnnClassifierMock extends Mock implements KnnClassifier {}

class KnnRegressorFactoryMock extends Mock implements KnnRegressorFactory {}

class KnnRegressorMock extends Mock implements KnnRegressor {}

class KnnSolverMock extends Mock implements KnnSolver {}

class KernelMock extends Mock implements Kernel {}

class LogisticRegressorMock extends Mock implements LogisticRegressor {}

class SoftmaxRegressorMock extends Mock implements SoftmaxRegressor {}

class SoftmaxRegressorFactoryMock extends Mock implements
    SoftmaxRegressorFactory {}

class ClassifierMock extends Mock implements Classifier {}

class PredictorMock extends Mock implements Predictor {}

class ClassifierAssessorMock extends Mock implements ClassifierAssessor {}

class LinearRegressorFactoryMock extends Mock
    implements LinearRegressorFactory {}

class LinearRegressorMock extends Mock implements LinearRegressor {}

class DecisionTreeClassifierFactoryMock extends Mock
    implements DecisionTreeClassifierFactory {}

class DecisionTreeClassifierMock extends Mock
    implements DecisionTreeClassifier {}

class LogisticRegressorFactoryMock extends Mock
    implements LogisticRegressorFactory {}

LearningRateGeneratorFactoryMock createLearningRateGeneratorFactoryMock(
    LearningRateGenerator generator) {
  final factory = LearningRateGeneratorFactoryMock();

  when(factory.fromType(any as LearningRateType)).thenReturn(generator);

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

  when(
    factory.fromType(
      any as InitialCoefficientsType,
      any as DType,
    ),
  ).thenReturn(generator);

  return factory;
}

CostFunctionFactory createCostFunctionFactoryMock(
    CostFunction costFunctionMock) {
  final costFunctionFactory = CostFunctionFactoryMock();

  when(
    costFunctionFactory.createByType(
      argThat(isNotNull) as CostFunctionType,
      linkFunction: anyNamed('linkFunction'),
      positiveLabel: anyNamed('positiveLabel'),
      negativeLabel: anyNamed('negativeLabel'),
  )).thenReturn(costFunctionMock);

  return costFunctionFactory;
}

ConvergenceDetectorFactory createConvergenceDetectorFactoryMock(
    ConvergenceDetector detector) {
  final convergenceDetectorFactory = ConvergenceDetectorFactoryMock();

  when(
    convergenceDetectorFactory.create(
      any as double,
      any as int,
    ),
  ).thenReturn(detector);

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

SplitIndicesProviderFactory createDataSplitterFactoryMock(SplitIndicesProvider dataSplitter) {
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
  when(factory.create(any, any, any, any, any, any, any))
      .thenReturn(classifier);
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
  when(factory.create(
    trainData: anyNamed('trainData'),
    targetName: anyNamed('targetName'),
    optimizerType: anyNamed('optimizerType'),
    iterationsLimit: anyNamed('iterationsLimit'),
    initialLearningRate: anyNamed('initialLearningRate'),
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate'),
    lambda: anyNamed('lambda'),
    regularizationType: anyNamed('regularizationType'),
    randomSeed: anyNamed('randomSeed'),
    batchSize: anyNamed('batchSize'),
    fitIntercept: anyNamed('fitIntercept'),
    interceptScale: anyNamed('interceptScale'),
    learningRateType: anyNamed('learningRateType'),
    isFittingDataNormalized: anyNamed('isFittingDataNormalized'),
    initialCoefficientsType: anyNamed('initialCoefficientsType'),
    initialCoefficients: anyNamed('initialCoefficients'),
    positiveLabel: anyNamed('positiveLabel'),
    negativeLabel: anyNamed('negativeLabel'),
    collectLearningData: anyNamed('collectLearningData'),
    probabilityThreshold: anyNamed('probabilityThreshold'),
    dtype: anyNamed('dtype'),
  ))
      .thenReturn(logisticRegressor);
  return factory;
}

SoftmaxRegressorFactory createSoftmaxRegressorFactoryMock(
    SoftmaxRegressor softmaxRegressor) {
  final factory = SoftmaxRegressorFactoryMock();
  when(factory.create(
    trainData: anyNamed('trainData'),
    targetNames: anyNamed('targetNames'),
    optimizerType: anyNamed('optimizerType'),
    iterationsLimit: anyNamed('iterationsLimit'),
    initialLearningRate: anyNamed('initialLearningRate'),
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate'),
    lambda: anyNamed('lambda'),
    regularizationType: anyNamed('regularizationType'),
    randomSeed: anyNamed('randomSeed'),
    batchSize: anyNamed('batchSize'),
    fitIntercept: anyNamed('fitIntercept'),
    interceptScale: anyNamed('interceptScale'),
    learningRateType: anyNamed('learningRateType'),
    isFittingDataNormalized: anyNamed('isFittingDataNormalized'),
    initialCoefficientsType: anyNamed('initialCoefficientsType'),
    initialCoefficients: anyNamed('initialCoefficients'),
    positiveLabel: anyNamed('positiveLabel'),
    negativeLabel: anyNamed('negativeLabel'),
    dtype: anyNamed('dtype'),
  ))
      .thenReturn(softmaxRegressor);
  return factory;
}

LinearRegressorFactory createLinearRegressorFactoryMock(
    LinearRegressor regressor) {
  final factory = LinearRegressorFactoryMock();
  when(factory.create(
    fittingData: anyNamed('fittingData'),
    targetName: anyNamed('targetName'),
    optimizerType: anyNamed('optimizerType'),
    iterationsLimit: anyNamed('iterationsLimit'),
    initialLearningRate: anyNamed('initialLearningRate'),
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate'),
    lambda: anyNamed('lambda'),
    regularizationType: anyNamed('regularizationType'),
    randomSeed: anyNamed('randomSeed'),
    batchSize: anyNamed('batchSize'),
    fitIntercept: anyNamed('fitIntercept'),
    interceptScale: anyNamed('interceptScale'),
    learningRateType: anyNamed('learningRateType'),
    isFittingDataNormalized: anyNamed('isFittingDataNormalized'),
    initialCoefficientsType: anyNamed('initialCoefficientsType'),
    initialCoefficients: anyNamed('initialCoefficients'),
    collectLearningData: anyNamed('collectLearningData'),
    dtype: anyNamed('dtype'),
  ))
      .thenReturn(regressor);
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

DistributionCalculatorFactory createDistributionCalculatorFactoryMock(
  DistributionCalculator calculator,
) {
  final factory = DistributionCalculatorFactoryMock();
  when(factory.create()).thenReturn(calculator);
  return factory;
}
