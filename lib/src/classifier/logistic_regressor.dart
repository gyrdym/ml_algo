import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/metric_type.dart';
import 'package:ml_algo/multinomial_type.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory_impl.dart';
import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/optimizer/gradient.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function_factory.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function_factory_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressor<T> implements LinearClassifier<T> {
  final GradientOptimizer<T> optimizer;
  final ScoreToProbLinkFunction<T> scoreToProbabilityLinkFn;
  final InterceptPreprocessor<T> interceptPreprocessor;
  final LabelsProcessor<T> labelsProcessor;

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

    // private arguments
    LabelsProcessorFactory labelsProcessorFactory = const LabelsProcessorFactoryImpl(),
    InterceptPreprocessorFactory interceptPreprocessorFactory = const InterceptPreprocessorFactoryImpl(),
    ScoreToProbLinkFunctionFactory scoreToProbFnFactory = const ScoreToProbLinkFunctionFactoryImpl(),
  }) :
    labelsProcessor = labelsProcessorFactory.create<T>(),
    interceptPreprocessor = interceptPreprocessorFactory.create<T>(scale: interceptScale),
    scoreToProbabilityLinkFn = scoreToProbFnFactory.create<T>(),
    optimizer = GradientOptimizer<T>(
        costFnType: CostFunctionType.logLikelihood,
        scoreToProbLink: scoreToProbFnFactory.create<T>(),
        learningRateType: learningRateType,
        initialWeightsType: initialWeightsType,
        initialLearningRate: learningRate,
        minCoefficientsUpdate: minWeightsUpdate,
        iterationLimit: iterationLimit,
        lambda: lambda,
        batchSize: batchSize,
        randomSeed: randomSeed,
    );

  MLMatrix<T> get weightsByClasses => _weightsByClasses;
  MLMatrix<T> _weightsByClasses;

  MLVector<T> get classLabels => _classLabels;
  MLVector<T> _classLabels;

  @override
  void fit(MLMatrix<T> features, MLVector<T> origLabels, {MLVector<T> initialWeights, bool isDataNormalized = false}) {
    _classLabels = origLabels.unique();
    final labelsAsList = _classLabels.toList();
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    final weights = List<MLVector<T>>.generate(labelsAsList.length, (int i) {
      final labels = labelsProcessor.makeLabelsOneVsAll(origLabels, labelsAsList[i]);
      return optimizer.findExtrema(processedFeatures, labels,
          initialWeights: initialWeights, arePointsNormalized: isDataNormalized, isMinimizingObjective: false);
    });
    _weightsByClasses = MLMatrix<T>.columns(weights);
  }

  @override
  double test(MLMatrix<T> features, MLVector<T> origLabels, MetricType metricType) {
    final evaluator = MetricFactory.createByType(metricType);
    final prediction = predictClasses(features);
    return evaluator.getError(prediction, origLabels);
  }

  @override
  MLMatrix<T> predictProbabilities(MLMatrix<T> features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    return _predictProbabilities(processedFeatures);
  }

  @override
  MLVector<T> predictClasses(MLMatrix<T> features) {
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    final distributions = _predictProbabilities(processedFeatures);
    final classes = List<double>(processedFeatures.rowsNum);
    for (int i = 0; i < distributions.rowsNum; i++) {
      final probabilities = distributions.getRow(i);
      classes[i] = probabilities.toList().indexOf(probabilities.max()) * 1.0;
    }
    return MLVector<T>.from(classes);
  }

  MLMatrix<T> _predictProbabilities(MLMatrix<T> processedFeatures) {
    final distributions = List<MLVector<T>>(_weightsByClasses.columnsNum);
    for (int i = 0; i < _weightsByClasses.columnsNum; i++) {
      final scores = (processedFeatures * _weightsByClasses.getColumn(i)).toVector();
      distributions[i] = scores.fastMap((T el, int startOffset, int endOffset) =>
          scoreToProbabilityLinkFn(el));
    }
    return MLMatrix<T>.columns(distributions);
  }
}
