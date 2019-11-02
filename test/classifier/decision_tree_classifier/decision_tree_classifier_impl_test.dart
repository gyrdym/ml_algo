import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/tree_solver/tree_solver.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label.dart';
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
    test('should call appropriate method from `solver` when making '
        'classes prediction for nominal class labels', () {
      final sample1 = Vector.fromList([1, 2, 3]);
      final sample2 = Vector.fromList([10, 20, 30]);
      final sample3 = Vector.fromList([100, 200, 300]);

      final features = Matrix.fromRows([
        sample1,
        sample2,
        sample3,
      ]);

      final label1 = 0.0;
      final label2 = 1.0;
      final label3 = 2.0;

      final solverMock = createSolver({
        sample1: TreeLeafLabel(label1),
        sample2: TreeLeafLabel(label2),
        sample3: TreeLeafLabel(label3),
      });

      final classifier = DecisionTreeClassifierImpl(solverMock, 'class_name',
          DType.float32);
      final predictedClasses = classifier.predict(
        DataFrame.fromMatrix(features),
      );

      expect(predictedClasses.header, equals(['class_name']));

      expect(predictedClasses.toMatrix(), equals([
        [label1],
        [label2],
        [label3],
      ]));
    });

    test('should return an empty matrix if input features matrix is '
        'empty', () {
      final solverMock = TreeSolverMock();
      final classifier = DecisionTreeClassifierImpl(solverMock, 'class_name',
          DType.float32);
      final predictedClasses = classifier.predict(DataFrame([<num>[]]));

      expect(predictedClasses.header, isEmpty);
      expect(predictedClasses.toMatrix(), isEmpty);
    });

    test('should call appropriate method from `solver` when making '
        'classes prediction for nominal class probabilities', () {
      final sample1 = Vector.fromList([1, 2, 3]);
      final sample2 = Vector.fromList([10, 20, 30]);
      final sample3 = Vector.fromList([100, 200, 300]);

      final features = Matrix.fromRows([
        sample1,
        sample2,
        sample3,
      ]);

      final label1 = TreeLeafLabel(0, probability: 0.7);
      final label2 = TreeLeafLabel(1, probability: 0.55);
      final label3 = TreeLeafLabel(2, probability: 0.5);

      final solverMock = createSolver({
        sample1: label1,
        sample2: label2,
        sample3: label3,
      });

      final classifier = DecisionTreeClassifierImpl(solverMock, 'class_name',
          DType.float32);
      final predictedLabels = classifier.predictProbabilities(
        DataFrame.fromMatrix(features),
      );

      expect(predictedLabels.header, equals(['class_name']));
      expect(
          predictedLabels.toMatrix(),
          iterable2dAlmostEqualTo([
            [label1.probability],
            [label2.probability],
            [label3.probability],
          ]),
      );
    });
  });
}

TreeSolver createSolver(Map<Vector, TreeLeafLabel> samples) {
  final solverMock = TreeSolverMock();
  samples.forEach((sample, leafLabel) =>
    when(solverMock.getLabelForSample(sample)).thenReturn(leafLabel));
  return solverMock;
}
