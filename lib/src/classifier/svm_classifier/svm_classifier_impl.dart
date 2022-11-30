import 'dart:io';

import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/svm_classifier/svm_classifier.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/linear_optimizer/svm_optimizer.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/src/data_frame/data_frame.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class SVMClassifierImpl with LinearClassifierMixin, AssessableClassifierMixin, SerializableMixin implements SVMClassifier {
  SVMClassifierImpl(
    DataFrame trainData,
    String targetName, {
    DType dtype = DType.float32,
    num learningRate = 1e-4,
    int iterationLimit = 100,
    bool fitIntercept = true,
    num interceptScale = 1,
    num negativeLabel = 0,
    num positiveLabel = 1,
  })  : dtype = dtype,
        learningRate = learningRate,
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        negativeLabel = negativeLabel,
        positiveLabel = positiveLabel {
    final splits = featuresTargetSplit(trainData, targetNames: [targetName]);
    final features = splits.first.toMatrix(dtype);
    final labels = splits.last.toMatrix(dtype);

    coefficientsByClasses = SVMOptimizer(
            features:
                addInterceptIf(fitIntercept, features, interceptScale, dtype),
            labels: labels,
            learningRate: learningRate,
            iterationLimit: iterationLimit,
            dtype: dtype)
        .findExtrema(isMinimizingObjective: true);
  }

  @override
  late Matrix coefficientsByClasses;

  @override
  final DType dtype;

  @override
  final bool fitIntercept;

  @override
  final num interceptScale;

  @override
  late LinkFunction linkFunction;

  @override
  final num negativeLabel;

  @override
  final num positiveLabel;

  final num learningRate;

  @override
  DataFrame predict(DataFrame testFeatures) {
    final predictedLabels = getProbabilitiesMatrix(testFeatures).mapColumns(
          (column) => column.mapToVector((probability) =>
      probability >= .5
          ? positiveLabel.toDouble()
          : negativeLabel.toDouble()),
    );

    return DataFrame.fromMatrix(
      predictedLabels,
      header: targetNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame testFeatures) {
    // TODO: implement predictProbabilities
    throw UnimplementedError();
  }

  @override
  SVMClassifier retrain(DataFrame data) {
    // TODO: implement retrain
    throw UnimplementedError();
  }

  @override
  Future<File> saveAsJson(String filePath) {
    // TODO: implement saveAsJson
    throw UnimplementedError();
  }

  @override
  // TODO: implement schemaVersion
  int? get schemaVersion => throw UnimplementedError();

  @override
  // TODO: implement targetNames
  Iterable<String> get targetNames => throw UnimplementedError();

  @override
  Map<String, dynamic> toJson() {
    // TODO: implement toJson
    throw UnimplementedError();
  }
}
