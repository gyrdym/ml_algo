import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory_impl.dart';
import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_factory_impl.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory_impl.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class SoftMaxRegressor implements LinearClassifier {
  final Type dtype;
  final Optimizer optimizer;
  final InterceptPreprocessor interceptPreprocessor;
  final LabelsProcessor labelsProcessor;
  final LinkFunction linkFunction;

  SoftMaxRegressor({
    // public arguments
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double initialLearningRate = DefaultParameterValues.initialLearningRate,
    double minWeightsUpdate = DefaultParameterValues.minWeightsUpdate,
    double lambda,
    int randomSeed,
    int batchSize = 1,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    OptimizerType optimizer = OptimizerType.gradientDescent,
    GradientType gradientType = GradientType.stochastic,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    LinkFunctionType linkFunctionType = LinkFunctionType.logit,
    this.dtype = DefaultParameterValues.dtype,

    // private arguments
    LabelsProcessorFactory labelsProcessorFactory =
        const LabelsProcessorFactoryImpl(),
    InterceptPreprocessorFactory interceptPreprocessorFactory =
        const InterceptPreprocessorFactoryImpl(),
    LinkFunctionFactory linkFunctionFactory = const LinkFunctionFactoryImpl(),
    OptimizerFactory optimizerFactory = const OptimizerFactoryImpl(),
    BatchSizeCalculator batchSizeCalculator = const BatchSizeCalculatorImpl(),
  })  : labelsProcessor = labelsProcessorFactory.create(dtype),
        interceptPreprocessor = interceptPreprocessorFactory.create(dtype,
            scale: fitIntercept ? interceptScale : 0.0),
        linkFunction = linkFunctionFactory.fromType(linkFunctionType, dtype),
        optimizer = optimizerFactory.fromType(
          optimizer,
          dtype: dtype,
          costFunctionType: CostFunctionType.logLikelihood,
          linkFunctionType: linkFunctionType,
          learningRateType: learningRateType,
          initialWeightsType: initialWeightsType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minWeightsUpdate,
          iterationLimit: iterationsLimit,
          lambda: lambda,
          batchSize: gradientType != null
              ? batchSizeCalculator.calculate(gradientType, batchSize)
              : null,
          randomSeed: randomSeed,
        );

  @override
  MLVector get weights => null;

  @override
  Map<double, MLVector> get weightsByClasses => _weightsByClasses;
  Map<double, MLVector> _weightsByClasses;

  @override
  List<double> get classLabels => _classLabels;
  List<double> _classLabels;

  @override
  void fit(MLMatrix features, MLVector labels,
      {MLVector initialWeights, bool isDataNormalized = false}) {
    _classLabels = labels.unique().toList();
    final labelsAsList = _classLabels.toList();
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    _weightsByClasses = Map<double, MLVector>.fromIterable(
      labelsAsList,
      key: (dynamic label) => label as double,
      value: (dynamic label) => _fitBinaryClassifier(processedFeatures, labels,
          label as double, initialWeights, isDataNormalized),
    );
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
    final numOfObservations = _weightsByClasses.length;
    final distributions = List<MLVector>(numOfObservations);
    int i = 0;
    _weightsByClasses.forEach((double label, MLVector weights) {
      final scores = (processedFeatures * weights).toVector();
      distributions[i++] = linkFunction.linkScoresToProbs(scores);
    });
    return MLMatrix.columns(distributions, dtype: dtype);
  }

  MLVector _fitBinaryClassifier(MLMatrix features, MLVector labels,
      double targetLabel, MLVector initialWeights, bool arePointsNormalized) {
    final binaryLabels =
        labelsProcessor.makeLabelsOneVsAll(labels, targetLabel);
    return optimizer.findExtrema(features, binaryLabels,
        initialWeights: initialWeights,
        arePointsNormalized: arePointsNormalized,
        isMinimizingObjective: false);
  }
}
