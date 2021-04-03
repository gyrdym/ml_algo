import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../mocks.dart';

void main() {
  group('TreeTrainerFactoryImpl', () {
    late TreeLeafDetector leafDetectorMock;
    late TreeLeafDetectorFactory leafDetectorFactoryMock;
    late TreeLeafLabelFactory leafLabelFactoryMock;
    late TreeLeafLabelFactoryFactory leafLabelFactoryFactoryMock;
    late TreeSplitSelector splitSelectorMock;
    late TreeSplitSelectorFactory splitSelectorFactoryMock;
    late TreeTrainerFactory factory;

    final type = TreeTrainerType.decision;
    final targetName = 'target';
    final header = ['feature_1', 'feature_2', 'feature_3', 'feature_4',
      targetName];
    final data = DataFrame([
      header,
      [1, 2, 3, 4, 100],
    ]);
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

      when(
        leafDetectorMock.isLeaf(
          any as Matrix,
          any as int,
          any as Iterable<int>,
          any as int,
        ),
      ).thenReturn(true);

      leafLabelFactoryMock = TreeLeafLabelFactoryMock();
      leafLabelFactoryFactoryMock = createTreeLeafLabelFactoryFactoryMock(
          leafLabelFactoryMock);

      splitSelectorMock = TreeSplitSelectorMock();
      splitSelectorFactoryMock = createTreeSplitSelectorFactoryMock(
          splitSelectorMock);

      factory = TreeTrainerFactoryImpl(leafDetectorFactoryMock,
          leafLabelFactoryFactoryMock, splitSelectorFactoryMock);

      factory.createByType(
        type,
        data,
        targetName,
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
