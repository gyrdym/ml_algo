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
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
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
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_type.dart';
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
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import './mocks.mocks.dart';

class MockEncoder extends Mock implements Encoder {
  @override
  DataFrame process(DataFrame? input) =>
      super.noSuchMethod(
        Invocation.method(#process, [input]),
//        returnValue: DataFrame([]),
      ) as DataFrame;
}

class EncoderFactoryMock extends Mock {
  Encoder create(DataFrame? data, Iterable<String>? targetNames);
}

class FeatureTargetSplitterMock extends Mock {
  Iterable<DataFrame> split(DataFrame? samples, {
    Iterable<String>? targetNames,
    Iterable<int>? targetIndices,
  });
}

class ClassLabelsNormalizerMock extends Mock {
  Matrix normalize(Matrix classLabels, num positiveLabel, num negativeLabel);
}

class LearningRateGeneratorMock extends Mock implements
    LearningRateGenerator {}

class LinearOptimizerFactoryMock extends Mock
    implements LinearOptimizerFactory {}

class LinearOptimizerMock extends Mock implements LinearOptimizer {}

class ConvergenceDetectorFactoryMock extends Mock
    implements ConvergenceDetectorFactory {}

class DataSplitterMock extends Mock implements SplitIndicesProvider {}

class DataSplitterFactoryMock extends Mock implements SplitIndicesProviderFactory {}

class AssessableMock extends Mock implements Assessable {}

class TreeLeafLabelFactoryFactoryMock extends Mock implements
    TreeLeafLabelFactoryFactory {}

class TreeSplitSelectorMock extends Mock implements TreeSplitSelector {}

class TreeSplitSelectorFactoryMock extends Mock implements
    TreeSplitSelectorFactory {}

class TreeNodeMock extends Mock implements TreeNode {}

class TreeSolverMock extends Mock implements DecisionTreeTrainer {}

class KnnClassifierMock extends Mock implements KnnClassifier {}

class KnnRegressorFactoryMock extends Mock implements KnnRegressorFactory {}

class KnnRegressorMock extends Mock implements KnnRegressor {}

class LogisticRegressorMock extends Mock implements LogisticRegressor {}

class SoftmaxRegressorMock extends Mock implements SoftmaxRegressor {}

class SoftmaxRegressorFactoryMock extends Mock implements
    SoftmaxRegressorFactory {}

class PredictorMock extends Mock implements Predictor {}

class LinearRegressorFactoryMock extends Mock
    implements LinearRegressorFactory {}

class LinearRegressorMock extends Mock implements LinearRegressor {}

class DecisionTreeClassifierMock extends Mock
    implements DecisionTreeClassifier {}

class LogisticRegressorFactoryMock extends Mock
    implements LogisticRegressorFactory {}

MockLearningRateGeneratorFactory createLearningRateGeneratorFactoryMock(
    LearningRateGenerator generator) {
  final factory = MockLearningRateGeneratorFactory();

  when(
    factory.fromType(any),
  ).thenReturn(generator);

  return factory;
}

RandomizerFactory createRandomizerFactoryMock(Randomizer randomizer) {
  final factory = MockRandomizerFactory();

  when(
    factory.create(any),
  ).thenReturn(randomizer);

  return factory;
}

InitialCoefficientsGeneratorFactory createInitialWeightsGeneratorFactoryMock(
    InitialCoefficientsGenerator generator) {
  final factory = MockInitialCoefficientsGeneratorFactory();

  when(
    factory.fromType(
      any,
      any,
    ),
  ).thenReturn(generator);

  return factory;
}

CostFunctionFactory createCostFunctionFactoryMock(
    CostFunction costFunctionMock) {
  final costFunctionFactory = MockCostFunctionFactory();

  when(
    costFunctionFactory.createByType(
      argThat(isNotNull),
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
    any as LinearOptimizerType,
    any as Matrix,
    any as Matrix,
    dtype: anyNamed('dtype') as DType,
    costFunction: anyNamed('costFunction') as CostFunction,
    learningRateType: anyNamed('learningRateType') as LearningRateType,
    initialCoefficientsType: anyNamed('initialCoefficientsType') as InitialCoefficientsType,
    initialLearningRate: anyNamed('initialLearningRate') as double,
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate') as double,
    iterationLimit: anyNamed('iterationLimit') as int,
    lambda: anyNamed('lambda') as double,
    regularizationType: anyNamed('regularizationType') as RegularizationType,
    batchSize: anyNamed('batchSize') as int,
    randomSeed: anyNamed('randomSeed') as int,
    isFittingDataNormalized: anyNamed('isFittingDataNormalized') as bool,
  )).thenReturn(optimizer);

  return factory;
}

SplitIndicesProviderFactory createDataSplitterFactoryMock(SplitIndicesProvider dataSplitter) {
  final factory = DataSplitterFactoryMock();

  when(factory.createByType(
    any as SplitIndicesProviderType,
    numberOfFolds: anyNamed('numberOfFolds'),
    p: anyNamed('p'),
  )).thenReturn(dataSplitter);

  return factory;
}

KernelFactory createKernelFactoryMock(Kernel kernel) {
  final factory = MockKernelFactory();

  when(factory.createByType(
    any as KernelType,
  )).thenReturn(kernel);

  return factory;
}

KnnSolverFactory createKnnSolverFactoryMock(KnnSolver solver) {
  final factory = MockKnnSolverFactory();

  when(
    factory.create(
      any,
      any,
      any,
      any,
      any,
    ),
  ).thenReturn(solver);

  return factory;
}

KnnClassifierFactory createKnnClassifierFactoryMock(KnnClassifier classifier) {
  final factory = MockKnnClassifierFactory();

  when(
    factory.create(
      any,
      any,
      any,
      any,
      any,
      any,
      any,
    ),
  ).thenReturn(classifier);

  return factory;
}

KnnRegressorFactory createKnnRegressorFactoryMock(KnnRegressor regressor) {
  final factory = KnnRegressorFactoryMock();

  when(factory.create(
    any as DataFrame,
    any as String,
    any as int,
    any as KernelType,
    any as Distance,
    any as DType,
  )).thenReturn(regressor);

  return factory;
}

LogisticRegressorFactory createLogisticRegressorFactoryMock(
    LogisticRegressor logisticRegressor) {
  final factory = LogisticRegressorFactoryMock();

  when(factory.create(
    trainData: anyNamed('trainData') as DataFrame,
    targetName: anyNamed('targetName') as String,
    optimizerType: anyNamed('optimizerType') as LinearOptimizerType,
    iterationsLimit: anyNamed('iterationsLimit') as int,
    initialLearningRate: anyNamed('initialLearningRate') as double,
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate') as double,
    lambda: anyNamed('lambda') as double,
    regularizationType: anyNamed('regularizationType') as RegularizationType,
    randomSeed: anyNamed('randomSeed') as int,
    batchSize: anyNamed('batchSize') as int,
    fitIntercept: anyNamed('fitIntercept') as bool,
    interceptScale: anyNamed('interceptScale') as double,
    learningRateType: anyNamed('learningRateType') as LearningRateType,
    isFittingDataNormalized: anyNamed('isFittingDataNormalized') as bool,
    initialCoefficientsType: anyNamed('initialCoefficientsType') as InitialCoefficientsType,
    initialCoefficients: anyNamed('initialCoefficients'),
    positiveLabel: anyNamed('positiveLabel') as num,
    negativeLabel: anyNamed('negativeLabel') as num,
    collectLearningData: anyNamed('collectLearningData') as bool,
    probabilityThreshold: anyNamed('probabilityThreshold') as double,
    dtype: anyNamed('dtype') as DType,
  ))
      .thenReturn(logisticRegressor);

  return factory;
}

SoftmaxRegressorFactory createSoftmaxRegressorFactoryMock(
    SoftmaxRegressor softmaxRegressor) {
  final factory = SoftmaxRegressorFactoryMock();

  when(factory.create(
    trainData: anyNamed('trainData') as DataFrame,
    targetNames: anyNamed('targetNames') as Iterable<String>,
    optimizerType: anyNamed('optimizerType') as LinearOptimizerType,
    iterationsLimit: anyNamed('iterationsLimit') as int,
    initialLearningRate: anyNamed('initialLearningRate') as double,
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate') as double,
    lambda: anyNamed('lambda') as double,
    regularizationType: anyNamed('regularizationType') as RegularizationType,
    randomSeed: anyNamed('randomSeed') as int,
    batchSize: anyNamed('batchSize') as int,
    fitIntercept: anyNamed('fitIntercept') as bool,
    interceptScale: anyNamed('interceptScale') as double,
    learningRateType: anyNamed('learningRateType') as LearningRateType,
    isFittingDataNormalized: anyNamed('isFittingDataNormalized') as bool,
    initialCoefficientsType: anyNamed('initialCoefficientsType') as InitialCoefficientsType,
    initialCoefficients: anyNamed('initialCoefficients'),
    positiveLabel: anyNamed('positiveLabel') as num,
    negativeLabel: anyNamed('negativeLabel') as num,
    collectLearningData: anyNamed('collectLearningData') as bool,
    dtype: anyNamed('dtype') as DType,
  ))
      .thenReturn(softmaxRegressor);

  return factory;
}

LinearRegressorFactory createLinearRegressorFactoryMock(
    LinearRegressor regressor) {
  final factory = LinearRegressorFactoryMock();

  when(factory.create(
    fittingData: anyNamed('fittingData') as DataFrame,
    targetName: anyNamed('targetName') as String,
    optimizerType: anyNamed('optimizerType') as LinearOptimizerType,
    iterationsLimit: anyNamed('iterationsLimit') as int,
    initialLearningRate: anyNamed('initialLearningRate') as double,
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate') as double,
    lambda: anyNamed('lambda') as double,
    regularizationType: anyNamed('regularizationType') as RegularizationType,
    randomSeed: anyNamed('randomSeed'),
    batchSize: anyNamed('batchSize') as int,
    fitIntercept: anyNamed('fitIntercept') as bool,
    interceptScale: anyNamed('interceptScale') as double,
    learningRateType: anyNamed('learningRateType') as LearningRateType,
    isFittingDataNormalized: anyNamed('isFittingDataNormalized') as bool,
    initialCoefficientsType: anyNamed('initialCoefficientsType') as InitialCoefficientsType,
    initialCoefficients: anyNamed('initialCoefficients'),
    collectLearningData: anyNamed('collectLearningData') as bool,
    dtype: anyNamed('dtype') as DType,
  ))
      .thenReturn(regressor);

  return factory;
}

MockTreeSplitAssessorFactory createTreeSplitAssessorFactoryMock(
    TreeSplitAssessor splitAssessor) {
  final factory = MockTreeSplitAssessorFactory();

  when(
    factory.createByType(any),
  ).thenReturn(splitAssessor);

  return factory;
}

MockTreeSplitterFactory createTreeSplitterFactoryMock(TreeSplitter splitter) {
  final factory = MockTreeSplitterFactory();

  when(factory.createByType(
    any,
    any,
  )).thenReturn(splitter);

  return factory;
}

@GenerateMocks([NumericalTreeSplitterFactory])
NumericalTreeSplitterFactory createNumericalTreeSplitterFactoryMock(
  NumericalTreeSplitter splitter,
) {
  final factory = MockNumericalTreeSplitterFactory();

  when(factory.create()).thenReturn(splitter);

  return factory;
}

@GenerateMocks([NominalTreeSplitterFactory])
NominalTreeSplitterFactory createNominalTreeSplitterFactoryMock(
  NominalTreeSplitter splitter,
) {
  final factory = MockNominalTreeSplitterFactory();

  when(factory.create()).thenReturn(splitter);

  return factory;
}

TreeLeafDetectorFactory createTreeLeafDetectorFactoryMock(
    TreeLeafDetector leafDetector) {
  final factory = MockTreeLeafDetectorFactory();

  when(factory.create(
    any,
    any,
    any,
    any,
  )).thenReturn(leafDetector);

  return factory;
}

TreeLeafLabelFactoryFactory createTreeLeafLabelFactoryFactoryMock(
  TreeLeafLabelFactory leafLabelFactory,
) {
  final factory = TreeLeafLabelFactoryFactoryMock();

  when(factory.createByType(
    any as TreeLeafLabelFactoryType,
  )).thenReturn(leafLabelFactory);

  return factory;
}

TreeSplitSelectorFactory createTreeSplitSelectorFactoryMock(
    TreeSplitSelector splitSelector) {
  final factory = TreeSplitSelectorFactoryMock();

  when(factory.createByType(
    any as TreeSplitSelectorType,
    any as TreeSplitAssessorType,
    any as TreeSplitterType,
  )).thenReturn(splitSelector);

  return factory;
}

MockDistributionCalculatorFactory createDistributionCalculatorFactoryMock(
  DistributionCalculator calculator,
) {
  final factory = MockDistributionCalculatorFactory();

  when(factory.create()).thenReturn(calculator);

  return factory;
}

@GenerateMocks([
  TreeSplitAssessor,
  TreeSplitAssessorFactory,
  NumericalTreeSplitter,
  NominalTreeSplitter,
  TreeSplitter,
  TreeSplitterFactory,
  MetricFactory,
  Metric,
  RandomizerFactory,
  Randomizer,
  CostFunction,
  CostFunctionFactory,
  DecisionTreeClassifierFactory,
  LinkFunction,
  InitialCoefficientsGenerator,
  InitialCoefficientsGeneratorFactory,
  ConvergenceDetector,
  DistributionCalculator,
  DistributionCalculatorFactory,
  LearningRateGeneratorFactory,
  TreeLeafDetector,
  TreeLeafLabelFactory,
  TreeLeafDetectorFactory,
  ClassifierAssessor,
  KnnSolver,
  KnnSolverFactory,
  Kernel,
  KernelFactory,
  KnnClassifierFactory,
  Classifier,
])
void main() {}
