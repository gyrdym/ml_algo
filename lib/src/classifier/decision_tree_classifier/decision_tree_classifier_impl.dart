import 'dart:io';

import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_constants.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/create_tree_svg_markup.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/from_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/to_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'decision_tree_classifier_impl.g.dart';

const classLabelsWarningMessage =
    'There is no use in positive and negative class labels for decision tree classifier';

@JsonSerializable()
@DTypeJsonConverter()
class DecisionTreeClassifierImpl
    with AssessableClassifierMixin, SerializableMixin
    implements DecisionTreeClassifier {
  DecisionTreeClassifierImpl(
    this.minError,
    this.minSamplesCount,
    this.maxDepth,
    this.treeRootNode,
    this.targetColumnName,
    this.assessorType,
    this.dtype, {
    this.schemaVersion = decisionTreeClassifierJsonSchemaVersion,
  });

  factory DecisionTreeClassifierImpl.fromJson(Map<String, dynamic> json) =>
      _$DecisionTreeClassifierImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$DecisionTreeClassifierImplToJson(this);

  @override
  @JsonKey(name: decisionTreeClassifierMinErrorJsonKey)
  final num minError;

  @override
  @JsonKey(name: decisionTreeClassifierMinSamplesCountJsonKey)
  final int minSamplesCount;

  @override
  @JsonKey(name: decisionTreeClassifierMaxDepthJsonKey)
  final int maxDepth;

  @override
  @JsonKey(name: decisionTreeClassifierDTypeJsonKey)
  final DType dtype;

  @JsonKey(name: decisionTreeClassifierTargetColumnNameJsonKey)
  final String targetColumnName;

  @override
  Iterable<String> get targetNames => [targetColumnName];

  @JsonKey(name: decisionTreeClassifierTreeRootNodeJsonKey)
  final TreeNode treeRootNode;

  @override
  num get positiveLabel => double.nan;

  @override
  num get negativeLabel => double.nan;

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final int schemaVersion;

  @override
  @JsonKey(
      name: decisionTreeClassifierAssessorTypeJsonKey,
      toJson: toTreeAssessorTypeJson,
      fromJson: fromTreeAssessorTypeJson)
  final TreeAssessorType assessorType;

  @override
  DataFrame predict(DataFrame features) {
    final predictedLabels = features
        .toMatrix(dtype)
        .rows
        .map((sample) => _getLabelForSample(sample, treeRootNode));

    if (predictedLabels.isEmpty) {
      return DataFrame([<num>[]]);
    }

    final outcomeList =
        predictedLabels.map((label) => label.value).toList(growable: false);
    final outcomeVector = Vector.fromList(outcomeList, dtype: dtype);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomeVector], dtype: dtype),
      header: targetNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final sampleVectors = features.toMatrix(dtype).rows;
    final probabilities = sampleVectors
        .map((sample) => _getLabelForSample(sample, treeRootNode))
        .map((label) => label.probability)
        .toList(growable: false);
    final probabilitiesVector = Vector.fromList(
      probabilities,
      dtype: dtype,
    );
    final probabilitiesMatrixColumn = Matrix.fromColumns([
      probabilitiesVector,
    ], dtype: dtype);

    return DataFrame.fromMatrix(
      probabilitiesMatrixColumn,
      header: targetNames,
    );
  }

  TreeLeafLabel _getLabelForSample(Vector sample, TreeNode node) {
    if (node.isLeaf) {
      return node.label!;
    }

    for (final childNode in node.children!) {
      if (childNode.testSample(sample)) {
        return _getLabelForSample(sample, childNode);
      }
    }

    throw Exception('Given sample does not conform any splitting condition');
  }

  @override
  DecisionTreeClassifier retrain(DataFrame data) {
    return decisionTreeInjector.get<DecisionTreeClassifierFactory>().create(
          data,
          targetColumnName,
          dtype,
          minError,
          minSamplesCount,
          maxDepth,
          assessorType,
        );
  }

  @override
  Future<File> saveAsSvg(String filePath) async {
    final markup = createTreeSvgMarkup(treeRootNode);
    final file = await File(filePath).create(recursive: true);

    return file.writeAsString(markup);
  }
}
