import 'package:inject/inject.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_factory_impl.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor_impl.dart';
import 'package:ml_algo/src/services/class_label_normalizer/class_label_normalizer.dart';
import 'package:ml_algo/src/services/class_label_normalizer/class_label_normalizer_impl.dart';
import 'package:ml_algo/src/services/encoder_factory/encoder_factory.dart';
import 'package:ml_algo/src/services/encoder_factory/encoder_factory_impl.dart';
import 'package:ml_algo/src/services/feature_target_splitter/feature_target_splitter.dart';
import 'package:ml_algo/src/services/feature_target_splitter/feature_target_splitter_impl.dart';

@module
class ModelAssessorModule {
  static ModelAssessorModule getInstance() =>
      _instance ??= ModelAssessorModule();
  static ModelAssessorModule _instance;

  @provide
  @singleton
  MetricFactory provideMetricFactory() =>
      const MetricFactoryImpl();

  @provide
  @singleton
  EncoderFactory provideEncoderFactory() =>
      const EncoderFactoryImpl();

  @provide
  @singleton
  FeatureTargetSplitter provideFeaturesTargetSplitter() =>
      const FeatureTargetSplitterImpl();

  @provide
  ClassLabelNormalizer provideCLassLabelNormalizer() =>
      const ClassLabelNormalizerImpl();

  @provide
  @singleton
  ClassifierAssessor provideClassifierAssessor(
      MetricFactory metricFactory,
      EncoderFactory encoderFactory,
      FeatureTargetSplitter featuresTargetSplitter,
      ClassLabelNormalizer classLabelNormalizer,
  ) => ClassifierAssessorImpl(
    metricFactory,
    encoderFactory,
    featuresTargetSplitter,
    classLabelNormalizer,
  );
}
