import 'package:ml_algo/src/tree_solver/decision_tree_solver.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_solver/tree_solver.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_factory.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../mocks.dart';

void main() {
  group('TreeSolverFactoryImpl', () {
    TreeLeafDetector leafDetectorMock;
    TreeLeafDetectorFactory leafDetectorFactoryMock;

    TreeLeafLabelFactory leafLabelFactoryMock;
    TreeLeafLabelFactoryFactory leafLabelFactoryFactoryMock;

    TreeSplitSelector splitSelectorMock;
    TreeSplitSelectorFactory splitSelectorFactoryMock;

    TreeSolverFactory factory;
    TreeSolver solver;

    final type = TreeSolverType.decision;
    final samples = Matrix.fromList([
      [ 10,  20,  30,  40,  50],
      [-10, -20, -30, -40, -50],
    ]);
    final featureIndices = [0, 1, 2, 3];
    final targetIdx = 4;
    final featureToUniqueValues = {
      1: [20, -20],
    };
    final minErrorOnNode = 0.3;
    final minSamplesCount = 10;
    final maxDepth = 7;
    final assessorType = TreeSplitAssessorType.majority;
    final leafLabelFactoryType = TreeLeafLabelFactoryType.majority;
    final splitSelectorType = TreeSplitSelectorType.greedy;
    final splitAssessorType = TreeSplitAssessorType.majority;
    final splitterType = TreeSplitterType.greedy;

    setUp(() {
      leafDetectorMock = TreeLeafDetectorMock();
      leafDetectorFactoryMock = createTreeLeafDetectorFactoryMock(
          leafDetectorMock);

      when(leafDetectorMock.isLeaf(any, any, any, any)).thenReturn(true);

      leafLabelFactoryMock = TreeLeafLabelFactoryMock();
      leafLabelFactoryFactoryMock = createTreeLeafLabelFactoryFactoryMock(
          leafLabelFactoryMock);

      splitSelectorMock = TreeSplitSelectorMock();
      splitSelectorFactoryMock = createTreeSplitSelectorFactoryMock(
          splitSelectorMock);

      factory = TreeSolverFactoryImpl(leafDetectorFactoryMock,
          leafLabelFactoryFactoryMock, splitSelectorFactoryMock);

      solver = factory.createByType(
        type,
        samples,
        featureIndices,
        targetIdx,
        featureToUniqueValues,
        minErrorOnNode,
        minSamplesCount,
        maxDepth,
        assessorType,
        leafLabelFactoryType,
        splitSelectorType,
        splitAssessorType,
        splitterType,
      );
    });

    tearDown(() {
      reset(leafDetectorMock);
      reset(leafLabelFactoryMock);
      reset(splitSelectorFactoryMock);
      reset(leafDetectorFactoryMock);
      reset(leafLabelFactoryFactoryMock);
      reset(splitSelectorFactoryMock);
    });

    test('should create a DecisionTreeSolver instance', () {
      expect(solver, isA<DecisionTreeSolver>());
    });

    test('should call leaf detector factory while creating the instance', () {
      verify(leafDetectorFactoryMock.create(assessorType, minErrorOnNode,
          minSamplesCount, maxDepth)).called(1);
    });

    test('should call leaf label factory factory while creating the instance', () {
      verify(leafLabelFactoryFactoryMock.createByType(leafLabelFactoryType))
          .called(1);
    });

    test('should call split selector factory while creating the instance', () {
      verify(splitSelectorFactoryMock.createByType(
          splitSelectorType, assessorType, splitterType)).called(1);
    });
  });
}
