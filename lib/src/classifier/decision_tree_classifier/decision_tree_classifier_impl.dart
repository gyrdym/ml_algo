import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_serializable_field.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/common/serializing_rule/dtype_serializing_rule.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_serialize.dart' as tree_node;
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class DecisionTreeClassifierImpl
    with
        AssessablePredictorMixin, SerializableMixin
    implements
        DecisionTreeClassifier {

  DecisionTreeClassifierImpl(this._root, String className, this.dtype)
      : classNames = [className];

  @override
  final DType dtype;

  @override
  final List<String> classNames;

  final TreeNode _root;

  @override
  DataFrame predict(DataFrame features) {
    final predictedLabels = features
        .toMatrix(dtype)
        .rows
        .map((sample) => _getLabelForSample(sample, _root));

    if (predictedLabels.isEmpty) {
      return DataFrame([<num>[]]);
    }

    final outcomeList = predictedLabels
        .map((label) => label.value)
        .toList(growable: false);
    final outcomeVector = Vector.fromList(outcomeList, dtype: dtype);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomeVector], dtype: dtype),
      header: classNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final probabilities = Matrix.fromColumns([
      Vector.fromList(
        features
            .toMatrix(dtype)
            .rows
            .map((sample) => _getLabelForSample(sample, _root))
            .map((label) => label.probability)
            .toList(growable: false),
        dtype: dtype,
      ),
    ], dtype: dtype);

    return DataFrame.fromMatrix(
      probabilities,
      header: classNames,
    );
  }

  @override
  Map<String, dynamic> serialize() => <String, dynamic>{
    dtypeField: dtypeSerializingRule[dtype],
    classNamesField: classNames,
    rootNodeField: tree_node.serialize(_root),
  };

  TreeLeafLabel _getLabelForSample(Vector sample, TreeNode node) {
    if (node.isLeaf) {
      return node.label;
    }

    for (final childNode in node.children) {
      if (childNode.isSamplePassed(sample)) {
        return _getLabelForSample(sample, childNode);
      }
    };

    throw Exception('Given sample does not conform any splitting condition');
  }
}
