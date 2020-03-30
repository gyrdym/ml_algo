import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_serializable_field.dart';
import 'package:ml_algo/src/common/serializable/primitive_serializer.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/common/serializable/serializer.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class DecisionTreeClassifierImpl
    with
        AssessablePredictorMixin, SerializableMixin
    implements
        DecisionTreeClassifier {

  DecisionTreeClassifierImpl(
      this._root,
      String className,
      this.dtype,
      this._dtypeSerializer,
      this._treeNodeSerializer,
  ) : classNames = [className];

  @override
  final DType dtype;

  @override
  final List<String> classNames;

  final PrimitiveSerializer<DType> _dtypeSerializer;

  final Serializer<TreeNode> _treeNodeSerializer;

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
    dtypeField: _dtypeSerializer.serialize(dtype),
    classNamesField: classNames,
    rootNodeField: _treeNodeSerializer.serialize(_root),
  };

  TreeLeafLabel _getLabelForSample(Vector sample, TreeNode node) {
    if (node.isLeaf) {
      return node.label;
    }

    for (final childNode in node.children) {
      if (childNode.isSamplePassed(sample) == null) {
        print(sample);
      }
      if (childNode.isSamplePassed(sample)) {
        return _getLabelForSample(sample, childNode);
      }
    }

    throw Exception('Given sample does not conform any splitting condition');
  }
}
