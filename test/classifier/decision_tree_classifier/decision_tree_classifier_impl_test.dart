import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
import 'package:ml_algo/src/common/dtype_serializer/dtype_encoded_values.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('DecisionTreeClassifierImpl', () {
    final sample1 = Vector.fromList([1, 2, 3]);
    final sample2 = Vector.fromList([10, 20, 30]);
    final sample3 = Vector.fromList([100, 200, 300]);

    final label1 = TreeLeafLabel(0, probability: 0.7);
    final label2 = TreeLeafLabel(1, probability: 0.55);
    final label3 = TreeLeafLabel(2, probability: 0.5);

    final targetColumnName = 'class_name';

    final features = Matrix.fromRows([
      sample1,
      sample2,
      sample3,
    ]);

    final rootNodeJson = {
      childrenJsonKey: <Map<String, dynamic>>[],
    };

    final classifierJson = {
      dTypeJsonKey: dTypeFloat32EncodedValue,
      targetColumnNameJsonKey: targetColumnName,
      treeRootNodeJsonKey: rootNodeJson,
    };

    final treeRootMock = createRootNodeMock({
      sample1: label1,
      sample2: label2,
      sample3: label3,
    }, rootNodeJson);

    DecisionTreeClassifierImpl classifier;

    setUp(() {
      classifier = DecisionTreeClassifierImpl(
        treeRootMock,
        targetColumnName,
        DType.float32,
      );
    });

    test('should return data frame with a correct header', () {
      final predictedLabels = classifier.predictProbabilities(
        DataFrame.fromMatrix(features),
      );
      expect(predictedLabels.header, equals(['class_name']));
    });

    test('should return a data frame with empty header if input matrix is '
        'empty', () {
      final predictedClasses = classifier.predict(DataFrame([<num>[]]));
      expect(predictedClasses.header, isEmpty);
    });

    test('should return data frame with empty matrix if input feature matrix is '
        'empty', () {
      final predictedClasses = classifier.predict(DataFrame([<num>[]]));
      expect(predictedClasses.toMatrix(), isEmpty);
    });

    test('should return data frame with probabilities for each class label', () {
      final predictedLabels = classifier.predictProbabilities(
        DataFrame.fromMatrix(features),
      );

      expect(
          predictedLabels.toMatrix(),
          iterable2dAlmostEqualTo([
            [label1.probability.toDouble()],
            [label2.probability.toDouble()],
            [label3.probability.toDouble()],
          ]),
      );
    });

    test('should serialize', () {
      final data = classifier.toJson();

      expect(data, equals(classifierJson));
      verify(treeRootMock.toJson()).called(1);
    });

    test('should be restored from json', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifierJson);

      expect(classifier.dtype, equals(DType.float32));
      expect(classifier.targetColumnName, equals(targetColumnName));
      expect(classifier.treeRootNode, isNotNull);
    });
  });
}

TreeNode createRootNodeMock(Map<Vector, TreeLeafLabel> samples,
    [Map<String, dynamic> jsonMock = const <String, dynamic>{}]) {

  final rootMock = TreeNodeMock();
  final children = <TreeNode>[];

  when(rootMock.isLeaf).thenReturn(false);

  samples.forEach((sample, leafLabel) {
    final node = TreeNodeMock();

    when(node.label).thenReturn(leafLabel);
    when(node.isLeaf).thenReturn(true);

    samples.forEach((otherSample, _) {
      when(node.isSamplePassed(otherSample)).thenReturn(sample == otherSample);
    });

    children.add(node);
  });
  
  when(rootMock.children).thenReturn(children);
  when(rootMock.toJson()).thenReturn(jsonMock);

  return rootMock;
}
