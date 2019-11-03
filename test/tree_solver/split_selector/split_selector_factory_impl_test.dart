import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_solver/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_type.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('TreeSplitSelectorFactoryImpl', () {
    final splitAssessorMock = TreeSplitAssessorMock();

    final splitAssessorFactoryMock =
      createTreeSplitAssessorFactoryMock(splitAssessorMock);

    final splitterMock = TreeSplitterMock();

    final splitterFactoryMock = createTreeSplitterFactoryMock(splitterMock);

    final factory = TreeSplitSelectorFactoryImpl(
        splitAssessorFactoryMock, splitterFactoryMock);

    setUp(() {
      reset(splitAssessorMock);
      reset(splitAssessorFactoryMock);
      reset(splitterMock);
      reset(splitterFactoryMock);
    });

    test('should create a GreedyTreeSplitSelector instance', () {
      final type = TreeSplitSelectorType.greedy;
      final assessorType = TreeSplitAssessorType.majority;
      final splitterType = TreeSplitterType.greedy;

      final splitSelector = factory.createByType(type, assessorType,
          splitterType);

      expect(splitSelector, isA<GreedyTreeSplitSelector>());
    });

    test('should throw an error if null passed as a split selector type', () {
      final assessorType = TreeSplitAssessorType.majority;
      final splitterType = TreeSplitterType.greedy;

      expect(() => factory.createByType(null, assessorType, splitterType),
          throwsUnsupportedError);
    });

    test('should call split assessor factory while creating the '
        'instance', () {

      final type = TreeSplitSelectorType.greedy;
      final assessorType = TreeSplitAssessorType.majority;
      final splitterType = TreeSplitterType.greedy;

      factory.createByType(type, assessorType, splitterType);

      verify(splitAssessorFactoryMock.createByType(assessorType))
          .called(1);
    });

    test('should call splitter factory while creating the instance', () {
      final type = TreeSplitSelectorType.greedy;
      final assessorType = TreeSplitAssessorType.majority;
      final splitterType = TreeSplitterType.greedy;

      factory.createByType(type, assessorType, splitterType);

      verify(splitterFactoryMock.createByType(splitterType, assessorType))
          .called(1);
    });
  });
}
