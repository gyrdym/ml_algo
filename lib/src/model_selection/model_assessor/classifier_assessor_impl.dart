import 'package:inject/inject.dart';
import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/metric/metric_constants.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor.dart';
import 'package:ml_algo/src/services/class_label_normalizer/class_label_normalizer.dart';
import 'package:ml_algo/src/services/encoder_factory/encoder_factory.dart';
import 'package:ml_algo/src/services/feature_target_splitter/feature_target_splitter.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

class ClassifierAssessorImpl implements ClassifierAssessor {
  @provide
  ClassifierAssessorImpl(
      this._metricFactory,
      this._encoderFactory,
      this._featuresTargetSplit,
      this._classLabelNormalizer,
  );

  final MetricFactory _metricFactory;
  final EncoderFactory _encoderFactory;
  final FeatureTargetSplitter _featuresTargetSplit;
  final ClassLabelNormalizer _classLabelNormalizer;

  @override
  double assess(
      Classifier classifier,
      MetricType metricType,
      DataFrame samples,
      ) {
    if (!classificationMetrics.contains(metricType)) {
      throw InvalidMetricTypeException(
          metricType, classificationMetrics);
    }

    final splits = _featuresTargetSplit.split(
      samples,
      targetNames: classifier.targetNames,
    ).toList();
    final featuresFrame = splits[0];
    final originalLabelsFrame = splits[1];
    final metric = _metricFactory
        .createByType(metricType);
    final labelEncoder = _encoderFactory.createOneHot(
        originalLabelsFrame,
        featureNames: originalLabelsFrame.header
    );
    final isTargetEncoded = classifier.targetNames.length > 1;
    final predictedLabels = !isTargetEncoded
        ? labelEncoder
        .process(classifier.predict(featuresFrame))
        .toMatrix(classifier.dtype)
        : classifier
        .predict(featuresFrame)
        .toMatrix(classifier.dtype);
    final originalLabels = !isTargetEncoded
        ? labelEncoder
        .process(originalLabelsFrame)
        .toMatrix(classifier.dtype)
        : originalLabelsFrame
        .toMatrix(classifier.dtype);
    final predefinedClassLabelsExist = classifier.negativeLabel != null
        && classifier.positiveLabel != null;
    final normalizedPredictedLabels = predefinedClassLabelsExist
        ? _classLabelNormalizer.normalize(
      predictedLabels,
      classifier.positiveLabel,
      classifier.negativeLabel,
    )
        : predictedLabels;
    final normalizedOriginalLabels = predefinedClassLabelsExist
        ? _classLabelNormalizer.normalize(
      originalLabels,
      classifier.positiveLabel,
      classifier.negativeLabel,
    )
        : originalLabels;

    return metric.getScore(
      normalizedPredictedLabels,
      normalizedOriginalLabels,
    );
  }
}
