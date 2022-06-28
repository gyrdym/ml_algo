import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_init_module.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/common/constants/default_parameters/classification.dart';
import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/constants/default_parameters/coordinate_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/linear_optimization.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

/// Softmax classification
///
/// Softmax regression is an algorithm that solves a multiclass classification
/// problem. The algorithm uses maximization of the passed
/// data likelihood (as well as
/// [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression).
/// In other words, the regressor iteratively tries to select coefficients
/// which makes combination of passed features and their coefficients most
/// likely. But instead of [Logit link function](https://en.wikipedia.org/wiki/Logit)
/// it uses [Softmax link function](https://en.wikipedia.org/wiki/Softmax_function)
/// that's why the algorithm has such a name.
///
/// Also it is worth to mention that the algorithm is a generalization of
/// [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression))
abstract class SoftmaxRegressor
    implements
        Assessable,
        Serializable,
        Retrainable<SoftmaxRegressor>,
        LinearClassifier {
  /// Parameters:
  ///
  /// [trainData] A [DataFrame] with observations which are used by the
  /// classifier to learn coefficients. Must contain [targetNames] columns.
  ///
  /// [targetNames] A collection of strings that serves as names for the
  /// target columns. A target column is a column that is containing class
  /// labels.
  ///
  /// [optimizerType] Defines an algorithm of optimization that will be used
  /// to find the best coefficients of log-likelihood cost function. Also
  /// defines a regularization type.
  ///
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of
  /// convergence in the optimization algorithm. Default value is `100`.
  ///
  /// [initialLearningRate] A value defining velocity of the convergence of the
  /// gradient descent optimizer. Default value is `1e-3`.
  ///
  /// [decay] The value meaning "speed" of learning rate decrease. Applicable only
  /// for [LearningRateType.timeBased], [LearningRateType.stepBased], and
  /// [LearningRateType.exponential] strategies
  ///
  /// [dropRate] The value that is used as a number of learning iterations after
  /// which the learning rate will be decreased. The value is applicable only for
  /// [LearningRateType.stepBased] learning rate; it will be omitted for other
  /// learning rate strategies
  ///
  /// [minCoefficientsUpdate] A minimum distance between coefficient vectors in
  /// two contiguous iterations. Uses as a condition of convergence in the
  /// optimization algorithm. If difference between the two vectors is small
  /// enough, there is no reason to continue fitting. Default value is `1e-12`.
  ///
  /// [lambda] A coefficient of regularization. Uses to prevent the regressor's
  /// overfitting. The more the value of [lambda], the more regular the
  /// coefficients are. Extremely large [lambda] may decrease the coefficients
  /// to nothing, otherwise too small [lambda] may be a cause of too large
  /// absolute values of the coefficients.
  ///
  /// [regularizationType] A way the coefficients of the classifier will be
  /// regularized to prevent overfitting of the model.
  ///
  /// [randomSeed] A seed that will be passed to a random value generator
  /// used by stochastic optimizers. Will be ignored, if the optimizer isn't
  /// stochastic. Remember, each time you run the stochastic regressor with the
  /// same parameters but with unspecified [randomSeed], you will receive
  /// different results. To avoid it, define [randomSeed]
  ///
  /// [batchSize] A size of data (in rows) that will be used per
  /// one fitting iteration. Applicable not for all optimizers. If gradient-based
  /// optimizer uses and If [batchSize] == `1`, stochastic mode will be
  /// activated; if `1` < [batchSize] < `total number of rows`, mini-batch mode
  /// will be activated; if [batchSize] == `total number of rows`, full-batch
  /// mode will be activated.
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `true`. Intercept in 2-dimensional space is a bias of the line (relative
  /// to X-axis) to be learned by the classifier.
  ///
  /// [interceptScale] A value defining a size of the intercept.
  ///
  /// [learningRateType] A value defining a strategy of the learning rate
  /// behaviour throughout the whole fitting process.
  ///
  /// [isFittingDataNormalized] Defines, whether the [trainData] normalized
  /// or not. Normalization should be performed column-wise. Normalized data
  /// may be required by some optimizers (e.g., for [LinearOptimizerType.coordinate])
  ///
  /// [initialCoefficientsType] Defines a type of coefficients (e.g. all zeroes,
  /// all random) that will be used in the very first iteration of optimization.
  /// By default, all the initial coefficients are equal to zeroes.
  /// If [initialCoefficients] are provided, the parameter will be ignored
  ///
  /// [initialCoefficients] Coefficients to be used in the bery first iteration of
  /// the optimization algorithm. [initialCoefficients] is a [Matrix], where the
  /// number of columns must be equal to the number of classes (or
  /// length of [targetNames]) and the number of rows must be equal to the
  /// number of features in [trainData]. In other words, every column of
  /// [initialCoefficients] matrix is a vector of coefficients of a certain
  /// class.
  ///
  /// [positiveLabel] A value that will be used for the positive class.
  /// By default, `1`.
  ///
  /// [negativeLabel] A value that will be used for the negative class.
  /// By default, `0`.
  ///
  /// [collectLearningData] Whether or not to collect learning data, for
  /// instance cost function value per each iteration. Affects performance much.
  /// If [collectLearningData] is true, one may access [costPerIteration]
  /// getter in order to evaluate learning process more thoroughly.
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can
  /// affect performance or accuracy of the computations. Default value is
  /// [DType.float32]
  factory SoftmaxRegressor(
    DataFrame trainData,
    List<String> targetNames, {
    LinearOptimizerType optimizerType = linearOptimizerTypeDefaultValue,
    int iterationsLimit = iterationLimitDefaultValue,
    double initialLearningRate = initialLearningRateDefaultValue,
    double decay = decayDefaultValue,
    int dropRate = dropRateDefaultValue,
    double minCoefficientsUpdate = minCoefficientsUpdateDefaultValue,
    double lambda = lambdaDefaultValue,
    int batchSize = batchSizeDefaultValue,
    bool fitIntercept = fitInterceptDefaultValue,
    double interceptScale = interceptScaleDefaultValue,
    LearningRateType learningRateType = learningRateTypeDefaultValue,
    bool isFittingDataNormalized = isFittingDataNormalizedDefaultValue,
    InitialCoefficientsType initialCoefficientsType =
        initialCoefficientsTypeDefaultValue,
    num positiveLabel = positiveLabelDefaultValue,
    num negativeLabel = negativeLabelDefaultValue,
    bool collectLearningData = collectLearningDataDefaultValue,
    DType dtype = dTypeDefaultValue,
    RegularizationType? regularizationType,
    Matrix? initialCoefficients,
    int? randomSeed,
  }) =>
      initSoftmaxRegressorModule().get<SoftmaxRegressorFactory>().create(
            trainData: trainData,
            targetNames: targetNames,
            optimizerType: optimizerType,
            iterationsLimit: iterationsLimit,
            initialLearningRate: initialLearningRate,
            decay: decay,
            dropRate: dropRate,
            minCoefficientsUpdate: minCoefficientsUpdate,
            lambda: lambda,
            regularizationType: regularizationType,
            randomSeed: randomSeed,
            batchSize: batchSize,
            fitIntercept: fitIntercept,
            interceptScale: interceptScale,
            learningRateType: learningRateType,
            isFittingDataNormalized: isFittingDataNormalized,
            initialCoefficientsType: initialCoefficientsType,
            initialCoefficients: initialCoefficients,
            positiveLabel: positiveLabel,
            negativeLabel: negativeLabel,
            collectLearningData: collectLearningData,
            dtype: dtype,
          );

  /// Restores previously fitted classifier instance from the [json]
  ///
  /// ````dart
  /// import 'dart:io';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// final data = <Iterable>[
  ///   ['feature 1', 'feature 2', 'feature 3', 'outcome 1', 'outcome 2']
  ///   [        5.0,         7.0,         6.0,         1.0,         0.0],
  ///   [        1.0,         2.0,         3.0,         0.0,         1.0],
  ///   [       10.0,        12.0,        31.0,         0.0,         1.0],
  ///   [        9.0,         8.0,         5.0,         0.0,         1.0],
  ///   [        4.0,         0.0,         1.0,         1.0,         0.0],
  /// ];
  /// final targetNames = ['outcome 1', 'outcome 2'];
  /// final samples = DataFrame(data, headerExists: true);
  /// final classifier = SoftmaxRegressor(
  ///   samples,
  ///   targetNames,
  ///   iterationsLimit: 2,
  ///   initialLearningRate: 1.0,
  ///   batchSize: 5,
  ///   fitIntercept: true,
  ///   interceptScale: 3.0,
  /// );
  ///
  /// final pathToFile = './classifier.json';
  ///
  /// await classifier.saveAsJson(pathToFile);
  ///
  /// final file = File(pathToFile);
  /// final json = await file.readAsString();
  /// final restoredClassifier = SoftmaxRegressor.fromJson(json);
  ///
  /// // here you can use previously fitted restored classifier to make
  /// // some prediction, e.g. via `restoredClassifier.predict(...)`;
  /// ````
  factory SoftmaxRegressor.fromJson(String json) => initSoftmaxRegressorModule()
      .get<SoftmaxRegressorFactory>()
      .fromJson(json);

  /// A linear optimization algorithm that was used
  /// to find the best coefficients of log-likelihood cost function. Also
  /// shows which regularization type (L1 or L2) was used to learn the model's
  /// coefficients.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  LinearOptimizerType get optimizerType;

  /// A number of fitting iterations that was used to learn the model's
  /// coefficients.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int get iterationsLimit;

  /// Initial learning rate value of chosen optimization algorithm
  ///
  /// The value is read-only, it's a hyperparameter of the model
  double get initialLearningRate;

  /// A value that was used for the learning rate decay
  ///
  /// The value is read-only, it's a hyperparameter of the model
  double get decay;

  /// A value that was used for the learning rate drop rate
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int get dropRate;

  /// A minimum distance between coefficient vectors in
  /// two contiguous iterations which was used to learn the model's
  /// coefficients.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  double get minCoefficientsUpdate;

  /// A coefficient of regularization
  ///
  /// The value is read-only, it's a hyperparameter of the model
  double get lambda;

  /// A way the coefficients of the classification were regularized during the
  /// model's coefficients learning process to prevent model overfitting.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  RegularizationType? get regularizationType;

  /// A seed that was passed to a random value generator used by a stochastic
  /// optimizer.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int? get randomSeed;

  /// A size of a batch of data (in rows) that was used in a single iteration
  /// of learning model's coefficients algorithm
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int get batchSize;

  /// Whether the fitting data was normalized or not prior to the model's
  /// coefficients learning
  ///
  /// The value is read-only, it's a hyperparameter of the model
  bool get isFittingDataNormalized;

  /// A type of a learning rate behaviour update strategy.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  LearningRateType get learningRateType;

  /// A coefficients generator type that was used by the chosen optimizer at
  /// the very first iteration of the model's coefficients learning algorithm.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  InitialCoefficientsType get initialCoefficientsType;

  /// Coefficients which were used at the very first learning iteration.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  Matrix? get initialCoefficients;

  /// Returns a list of cost values per each learning iteration. Returns null
  /// if the parameter `collectLearningData` of the default constructor is false
  List<num>? get costPerIteration;
}
