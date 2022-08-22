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
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_iterable_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
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
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';

import './mocks.mocks.dart';

class MockEncoder extends Mock implements Encoder {
  @override
  DataFrame process(DataFrame? input) {
    return super.noSuchMethod(
      Invocation.method(#process, [input]),
      returnValue: DataFrame([]),
    ) as DataFrame;
  }
}

class MockEncoderFactory extends Mock {
  Encoder create(DataFrame? data, Iterable<String>? targetNames) {
    return super.noSuchMethod(
      Invocation.method(#create, [data, targetNames]),
      returnValue: MockEncoder(),
    ) as MockEncoder;
  }
}

class MockFeatureTargetSplitter extends Mock {
  Iterable<DataFrame> split(
    DataFrame? samples, {
    Iterable<String>? targetNames,
    Iterable<int>? targetIndices,
  }) {
    return super.noSuchMethod(
      Invocation.method(#split, [
        samples
      ], {
        #targetNames: targetNames,
        #targetIndices: targetIndices,
      }),
      returnValue: <DataFrame>[],
    ) as Iterable<DataFrame>;
  }
}

class MockClassLabelsNormalizer extends Mock {
  Matrix normalize(
      Matrix? classLabels, num? positiveLabel, num? negativeLabel) {
    return super.noSuchMethod(
      Invocation.method(
          #normalize, [classLabels, positiveLabel, negativeLabel]),
      returnValue: Matrix.empty(),
    ) as Matrix;
  }
}

MockLearningRateIterableFactory createLearningRateGeneratorFactoryMock(
    Iterable<double> iterable) {
  final factory = MockLearningRateIterableFactory();

  when(
    factory.fromType(
      type: anyNamed('type'),
      initialValue: anyNamed('initialValue'),
      decay: anyNamed('decay'),
      iterationLimit: anyNamed('iterationLimit'),
    ),
  ).thenReturn(iterable);

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

MockCostFunctionFactory createCostFunctionFactoryMock(
    CostFunction costFunctionMock) {
  final costFunctionFactory = MockCostFunctionFactory();

  when(costFunctionFactory.createByType(
    any,
    linkFunction: anyNamed('linkFunction'),
    positiveLabel: anyNamed('positiveLabel'),
    negativeLabel: anyNamed('negativeLabel'),
    dtype: anyNamed('dtype'),
  )).thenReturn(costFunctionMock);

  return costFunctionFactory;
}

ConvergenceDetectorFactory createConvergenceDetectorFactoryMock(
    ConvergenceDetector detector) {
  final convergenceDetectorFactory = MockConvergenceDetectorFactory();

  when(
    convergenceDetectorFactory.create(
      any,
      any,
    ),
  ).thenReturn(detector);

  return convergenceDetectorFactory;
}

MockLinearOptimizerFactory createLinearOptimizerFactoryMock(
    LinearOptimizer optimizer) {
  final factory = MockLinearOptimizerFactory();

  when(factory.createByType(
    any,
    any,
    any,
    dtype: anyNamed('dtype'),
    costFunction: anyNamed('costFunction'),
    learningRateType: anyNamed('learningRateType'),
    initialCoefficientsType: anyNamed('initialCoefficientsType'),
    decay: anyNamed('decay'),
    dropRate: anyNamed('dropRate'),
    initialLearningRate: anyNamed('initialLearningRate'),
    minCoefficientsUpdate: anyNamed('minCoefficientsUpdate'),
    iterationLimit: anyNamed('iterationLimit'),
    lambda: anyNamed('lambda'),
    regularizationType: anyNamed('regularizationType'),
    batchSize: anyNamed('batchSize'),
    randomSeed: anyNamed('randomSeed'),
    isFittingDataNormalized: anyNamed('isFittingDataNormalized'),
  )).thenReturn(optimizer);

  return factory;
}

SplitIndicesProviderFactory createDataSplitterFactoryMock(
    SplitIndicesProvider dataSplitter) {
  final factory = MockSplitIndicesProviderFactory();

  when(factory.createByType(
    any,
    numberOfFolds: anyNamed('numberOfFolds'),
    p: anyNamed('p'),
  )).thenReturn(dataSplitter);

  return factory;
}

KernelFactory createKernelFactoryMock(Kernel kernel) {
  final factory = MockKernelFactory();

  when(
    factory.createByType(
      any,
    ),
  ).thenReturn(kernel);

  return factory;
}

MockKnnSolverFactory createKnnSolverFactoryMock(KnnSolver solver) {
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
  final factory = MockKnnRegressorFactory();

  when(
    factory.create(
      any,
      any,
      any,
      any,
      any,
      any,
    ),
  ).thenReturn(regressor);

  return factory;
}

LogisticRegressorFactory createLogisticRegressorFactoryMock(
    LogisticRegressor logisticRegressor) {
  final factory = MockLogisticRegressorFactory();

  when(
    factory.create(
      trainData: anyNamed('trainData'),
      targetName: anyNamed('targetName'),
      optimizerType: anyNamed('optimizerType'),
      iterationsLimit: anyNamed('iterationsLimit'),
      initialLearningRate: anyNamed('initialLearningRate'),
      decay: anyNamed('decay'),
      dropRate: anyNamed('dropRate'),
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
    ),
  ).thenReturn(logisticRegressor);

  return factory;
}

SoftmaxRegressorFactory createSoftmaxRegressorFactoryMock(
    SoftmaxRegressor softmaxRegressor) {
  final factory = MockSoftmaxRegressorFactory();

  when(
    factory.create(
      trainData: anyNamed('trainData'),
      targetNames: anyNamed('targetNames'),
      optimizerType: anyNamed('optimizerType'),
      iterationsLimit: anyNamed('iterationsLimit'),
      initialLearningRate: anyNamed('initialLearningRate'),
      decay: anyNamed('decay'),
      dropRate: anyNamed('dropRate'),
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
      dtype: anyNamed('dtype'),
    ),
  ).thenReturn(softmaxRegressor);

  return factory;
}

LinearRegressorFactory createLinearRegressorFactoryMock(
    LinearRegressor regressor) {
  final factory = MockLinearRegressorFactory();

  when(
    factory.create(
      fittingData: anyNamed('fittingData'),
      targetName: anyNamed('targetName'),
      optimizerType: anyNamed('optimizerType'),
      iterationsLimit: anyNamed('iterationsLimit'),
      initialLearningRate: anyNamed('initialLearningRate'),
      decay: anyNamed('decay'),
      dropRate: anyNamed('dropRate'),
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
    ),
  ).thenReturn(regressor);

  return factory;
}

MockTreeAssessorFactory createTreeSplitAssessorFactoryMock(
    TreeAssessor splitAssessor) {
  final factory = MockTreeAssessorFactory();

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
  final factory = MockTreeLeafLabelFactoryFactory();

  when(
    factory.createByType(
      any,
    ),
  ).thenReturn(leafLabelFactory);

  return factory;
}

TreeSplitSelectorFactory createTreeSplitSelectorFactoryMock(
    TreeSplitSelector splitSelector) {
  final factory = MockTreeSplitSelectorFactory();

  when(
    factory.createByType(
      any,
      any,
      any,
    ),
  ).thenReturn(splitSelector);

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
  TreeAssessor,
  TreeAssessorFactory,
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
  LearningRateIterableFactory,
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
  LinearOptimizerFactory,
  LinearOptimizer,
  ConvergenceDetectorFactory,
  SplitIndicesProvider,
  SplitIndicesProviderFactory,
  Assessable,
  TreeLeafLabelFactoryFactory,
  TreeSplitSelector,
  TreeSplitSelectorFactory,
  TreeNode,
  DecisionTreeTrainer,
  KnnClassifier,
  KnnRegressorFactory,
  KnnRegressor,
  LogisticRegressor,
  SoftmaxRegressor,
  SoftmaxRegressorFactory,
  Predictor,
  LinearRegressorFactory,
  LinearRegressor,
  DecisionTreeClassifier,
  LogisticRegressorFactory,
])
void main() {}
