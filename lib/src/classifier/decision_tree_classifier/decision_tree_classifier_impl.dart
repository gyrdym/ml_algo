import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/_helper/from_tree_node_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/_helper/tree_node_to_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/from_dtype_json.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'decision_tree_classifier_impl.g.dart';

@JsonSerializable()
class DecisionTreeClassifierImpl
    with
        AssessableClassifierMixin,
        SerializableMixin
    implements
        DecisionTreeClassifier {

  DecisionTreeClassifierImpl(
      this.treeRootNode,
      this.targetColumnName,
      this.dtype,
  );

  factory DecisionTreeClassifierImpl.fromJson(Map<String, dynamic> json) =>
      _$DecisionTreeClassifierImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$DecisionTreeClassifierImplToJson(this);

  @override
  @JsonKey(
    name: dTypeJsonKey,
    toJson: dTypeToJson,
    fromJson: fromDTypeJson,
  )
  final DType dtype;

  @JsonKey(name: targetColumnNameJsonKey)
  final String targetColumnName;

  @override
  Iterable<String> get targetNames => [targetColumnName];

  @JsonKey(
    name: treeRootNodeJsonKey,
    toJson: treeNodeToJson,
    fromJson: fromTreeNodeJson,
  )
  final TreeNode treeRootNode;

  @override
  @JsonKey(includeIfNull: false)
  final num positiveLabel = null;

  @override
  @JsonKey(includeIfNull: false)
  final num negativeLabel = null;

  @override
  DataFrame predict(DataFrame features) {
    final predictedLabels = features
        .toMatrix(dtype)
        .rows
        .map((sample) => _getLabelForSample(sample, treeRootNode));

    if (predictedLabels.isEmpty) {
      return DataFrame([<num>[]]);
    }

    final outcomeList = predictedLabels
        .map((label) => label.value)
        .toList(growable: false);
    final outcomeVector = Vector.fromList(outcomeList, dtype: dtype);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomeVector], dtype: dtype),
      header: targetNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final sampleVectors = features
        .toMatrix(dtype)
        .rows;
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
      return node.label;
    }

    for (final childNode in node.children) {
      if (childNode.isSamplePassed(sample)) {
        return _getLabelForSample(sample, childNode);
      }
    }

    throw Exception('Given sample does not conform any splitting condition');
  }
}
