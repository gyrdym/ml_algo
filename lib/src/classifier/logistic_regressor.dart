import 'dart:typed_data';

import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/metric_type.dart';
import 'package:ml_algo/multinomial_type.dart';
import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator.dart';
import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator_factory.dart';
import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator_factory_impl.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory_impl.dart';
import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressor implements LinearClassifier {
  final Type dtype;
  final Optimizer optimizer;
  final InterceptPreprocessor interceptPreprocessor;
  final LabelsProcessor labelsProcessor;
  final LabelsProbabilityCalculator probabilityCalculator;

  LogisticRegressor({
    // public arguments
    int iterationLimit,
    double learningRate,
    double minWeightsUpdate,
    double lambda,
    int batchSize = 1,
    int randomSeed,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    MultinomialType multinomialType = MultinomialType.oneVsAll,
    LearningRateType learningRateType = LearningRateType.decreasing,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    LinkFunctionType linkFunctionType = LinkFunctionType.logit,
    this.dtype = Float32x4,

    // private arguments
    LabelsProcessorFactory labelsProcessorFactory = const LabelsProcessorFactoryImpl(),
    InterceptPreprocessorFactory interceptPreprocessorFactory = const InterceptPreprocessorFactoryImpl(),
    LabelsProbabilityCalculatorFactory probabilityCalculatorFactory = const LabelsProbabilityCalculatorFactoryImpl(),
    OptimizerFactory optimizerFactory = const OptimizerFactoryImpl(),
  }) :
    labelsProcessor = labelsProcessorFactory.create(dtype),
    interceptPreprocessor = interceptPreprocessorFactory.create(dtype, scale: interceptScale),
    probabilityCalculator = probabilityCalculatorFactory.create(linkFunctionType, dtype),
    optimizer = optimizerFactory.gradient(
        costFnType: CostFunctionType.logLikelihood,
        linkFunctionType: linkFunctionType,
        learningRateType: learningRateType,
        initialWeightsType: initialWeightsType,
        initialLearningRate: learningRate,
        minCoefficientsUpdate: minWeightsUpdate,
        iterationLimit: iterationLimit,
        lambda: lambda,
        batchSize: batchSize,
        randomSeed: randomSeed,
    );

  @override
  MLVector get weights => null;

  MLMatrix get weightsByClasses => _weightsByClasses;
  MLMatrix _weightsByClasses;

  MLVector get classLabels => _classLabels;
  MLVector _classLabels;

  @override
  void fit(MLMatrix features, MLVector origLabels, {MLVector initialWeights, bool isDataNormalized = false}) {
    _classLabels = origLabels.unique();
    final labelsAsList = _classLabels.toList();
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    final weights = List<MLVector>.generate(labelsAsList.length, (int i) {
      final labels = labelsProcessor.makeLabelsOneVsAll(origLabels, labelsAsList[i]);
      return optimizer.findExtrema(processedFeatures, labels,
          initialWeights: initialWeights, arePointsNormalized: isDataNormalized, isMinimizingObjective: false);
    });
    _weightsByClasses = MLMatrix.columns(weights, dtype: dtype);
  }

  @override
  double test(MLMatrix features, MLVector origLabels, MetricType metricType) {
    final evaluator = MetricFactory.createByType(metricType);
    final prediction = predictClasses(features);
    return evaluator.getError(prediction, origLabels);
  }

  @override
  MLMatrix predictProbabilities(MLMatrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures);
  }

  @override
  MLVector predictClasses(MLMatrix features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    final distributions = _predictProbabilities(processedFeatures);
    final classes = List<double>(processedFeatures.rowsNum);
    for (int i = 0; i < distributions.rowsNum; i++) {
      final probabilities = distributions.getRow(i);
      classes[i] = probabilities.toList().indexOf(probabilities.max()) * 1.0;
    }
    return MLVector.from(classes, dtype: dtype);
  }

  MLMatrix _predictProbabilities(MLMatrix processedFeatures) {
    final distributions = List<MLVector>(_weightsByClasses.columnsNum);
    for (int i = 0; i < _weightsByClasses.columnsNum; i++) {
      final scores = (processedFeatures * _weightsByClasses.getColumn(i)).toVector();
      distributions[i] = probabilityCalculator.getProbabilities(scores);
    }
    return MLMatrix.columns(distributions, dtype: dtype);
  }
}
