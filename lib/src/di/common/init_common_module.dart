import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/di/dependency_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/extensions/get_it.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/helpers/features_target_split_interface.dart';
import 'package:ml_algo/src/helpers/normalize_class_labels.dart';
import 'package:ml_algo/src/helpers/normalize_class_labels_interface.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory_impl.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory_impl.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_factory_impl.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor.dart';
import 'package:ml_algo/src/model_selection/model_assessor/regressor_assessor.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

typedef EncoderFactory = Encoder Function(DataFrame, Iterable<String>);

void initCommonModule() {
  module
    ..registerSingletonIf<EncoderFactory>(
            (DataFrame data, Iterable<String> targetNames) =>
                Encoder.oneHot(data, featureNames: targetNames),
        instanceName: oneHotEncoderFactoryKey)

    ..registerSingletonIf<RandomizerFactory>(const RandomizerFactoryImpl())

    ..registerSingletonIf<FeaturesTargetSplit>(featuresTargetSplit)

    ..registerSingleton<MetricFactory>(const MetricFactoryImpl())

    ..registerSingletonIf<NormalizeClassLabels>(normalizeClassLabels)

    ..registerSingletonIf<LinearOptimizerFactory>(
        const LinearOptimizerFactoryImpl())

    ..registerSingletonIf<LearningRateGeneratorFactory>(
        const LearningRateGeneratorFactoryImpl())

    ..registerSingletonIf<InitialCoefficientsGeneratorFactory>(
        const InitialCoefficientsGeneratorFactoryImpl())

    ..registerSingletonIf<ConvergenceDetectorFactory>(
        const ConvergenceDetectorFactoryImpl())

    ..registerSingletonIf<CostFunctionFactory>(
        const CostFunctionFactoryImpl())
      
    ..registerSingletonWithDependencies<ModelAssessor<Classifier>>(
        () => ClassifierAssessor(
          module<MetricFactory>(),
          module<EncoderFactory>(instanceName: oneHotEncoderFactoryKey),
          module<FeaturesTargetSplit>(),
          module<NormalizeClassLabels>(),
        ),
        dependsOn: [
          MetricFactory,
          EncoderFactory,
          FeaturesTargetSplit,
          NormalizeClassLabels,
        ])

    ..registerSingletonWithDependencies<ModelAssessor<Predictor>>(
        () => RegressorAssessor(
          module.get<MetricFactory>(),
          module.get<FeaturesTargetSplit>(),
        ),
        dependsOn: [
          MetricFactory,
          FeaturesTargetSplit,
        ]);
}
