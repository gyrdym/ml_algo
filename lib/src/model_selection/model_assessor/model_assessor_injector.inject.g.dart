import 'model_assessor_injector.dart' as _i1;
import 'model_assessor_module.dart' as _i2;
import '../../metric/metric_factory.dart' as _i3;
import '../../services/encoder_factory/encoder_factory.dart' as _i4;
import '../../services/feature_target_splitter/feature_target_splitter.dart'
    as _i5;
import 'classifier_assessor.dart' as _i6;
import 'dart:async' as _i7;
import '../../services/class_label_normalizer/class_label_normalizer.dart'
    as _i8;

class ModelAssessorInjector$Injector implements _i1.ModelAssessorInjector {
  ModelAssessorInjector$Injector.fromModule(this._modelAssessorModule);

  final _i2.ModelAssessorModule _modelAssessorModule;

  _i3.MetricFactory _singletonMetricFactory;

  _i4.EncoderFactory _singletonEncoderFactory;

  _i5.FeatureTargetSplitter _singletonFeatureTargetSplitter;

  _i6.ClassifierAssessor _singletonClassifierAssessor;

  static _i7.Future<_i1.ModelAssessorInjector> create(
      _i2.ModelAssessorModule modelAssessorModule) async {
    final injector =
        ModelAssessorInjector$Injector.fromModule(modelAssessorModule);

    return injector;
  }

  _i6.ClassifierAssessor _createClassifierAssessor() =>
      _singletonClassifierAssessor ??=
          _modelAssessorModule.provideClassifierAssessor(
              _createMetricFactory(),
              _createEncoderFactory(),
              _createFeatureTargetSplitter(),
              _createClassLabelNormalizer());
  _i3.MetricFactory _createMetricFactory() =>
      _singletonMetricFactory ??= _modelAssessorModule.provideMetricFactory();
  _i4.EncoderFactory _createEncoderFactory() =>
      _singletonEncoderFactory ??= _modelAssessorModule.provideEncoderFactory();
  _i5.FeatureTargetSplitter _createFeatureTargetSplitter() =>
      _singletonFeatureTargetSplitter ??=
          _modelAssessorModule.provideFeaturesTargetSplitter();
  _i8.ClassLabelNormalizer _createClassLabelNormalizer() =>
      _modelAssessorModule.provideCLassLabelNormalizer();
  @override
  _i6.ClassifierAssessor getClassifierAssessor() => _createClassifierAssessor();
}
