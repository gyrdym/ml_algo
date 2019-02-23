import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory_impl.dart';
import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory_impl.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/regressor/gradient_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory_impl.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class SoftmaxRegressor implements LinearClassifier {
  final Type dtype;
  final Optimizer optimizer;
  final InterceptPreprocessor interceptPreprocessor;
  final LabelsProcessor labelsProcessor;
  final ScoreToProbMapper scoreToProbMapper;
  final CategoricalDataEncoder dataEncoder;

  SoftmaxRegressor({
    // public arguments
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double initialLearningRate = DefaultParameterValues.initialLearningRate,
    double minWeightsUpdate = DefaultParameterValues.minCoefficientsUpdate,
    double lambda,
    int randomSeed,
    int batchSize = 1,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    OptimizerType optimizer = OptimizerType.gradientDescent,
    GradientType gradientType = GradientType.stochastic,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    ScoreToProbMapperType scoreToProbMapperType = ScoreToProbMapperType.softmax,
    this.dtype = DefaultParameterValues.dtype,

    // private arguments
    LabelsProcessorFactory labelsProcessorFactory =
    const LabelsProcessorFactoryImpl(),
    InterceptPreprocessorFactory interceptPreprocessorFactory =
    const InterceptPreprocessorFactoryImpl(),
    ScoreToProbMapperFactory scoreToProbMapperFactory =
    const ScoreToProbMapperFactoryImpl(),
    OptimizerFactory optimizerFactory = const OptimizerFactoryImpl(),
    BatchSizeCalculator batchSizeCalculator = const BatchSizeCalculatorImpl(),
    CategoricalDataEncoderFactory categoricalDataEncoderFactory =
    const CategoricalDataEncoderFactory(),
  })
      : labelsProcessor = labelsProcessorFactory.create(dtype),
        interceptPreprocessor = interceptPreprocessorFactory.create(dtype,
            scale: fitIntercept ? interceptScale : 0.0),
        scoreToProbMapper =
        scoreToProbMapperFactory.fromType(scoreToProbMapperType, dtype),
        dataEncoder = categoricalDataEncoderFactory.oneHot(),
        optimizer = optimizerFactory.fromType(
          optimizer,
          dtype: dtype,
          costFunctionType: CostFunctionType.logLikelihood,
          scoreToProbMapperType: scoreToProbMapperType,
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

  MLMatrix _weights;

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
    final processedFeatures = interceptPreprocessor.addIntercept(features);
    _weights = _learnWeights(
        processedFeatures, labels, initialWeights, isDataNormalized);
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

  MLMatrix _predictProbabilities(MLMatrix processedFeatures) =>
      scoreToProbMapper.linkScoresToProbs(processedFeatures * _weights);

  MLMatrix _learnWeights(MLMatrix features, MLVector labels,
      MLVector initialWeights, bool arePointsNormalized) {
    final oneHotEncodedLabels = dataEncoder.encodeAll(labels);
    return optimizer
        .findExtrema(features, oneHotEncodedLabels,
        initialWeights: initialWeights != null
            ? MLMatrix.rows([initialWeights]) : null,
        arePointsNormalized: arePointsNormalized,
        isMinimizingObjective: false);
  }
}
