import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('TreeSplitSelectorFactoryImpl', () {
    late TreeSplitAssessor splitAssessorMock;
    late MockTreeSplitAssessorFactory splitAssessorFactoryMock;
    late MockTreeSplitter splitterMock;
    late MockTreeSplitterFactory splitterFactoryMock;
    late TreeSplitSelectorFactoryImpl factory;

    setUp(() {
      splitAssessorMock = MockTreeSplitAssessor();
      splitAssessorFactoryMock =
          createTreeSplitAssessorFactoryMock(splitAssessorMock);
      splitterMock = MockTreeSplitter();
      splitterFactoryMock = createTreeSplitterFactoryMock(splitterMock);
      factory = TreeSplitSelectorFactoryImpl(
          splitAssessorFactoryMock, splitterFactoryMock);
    });

    tearDown(() {
      reset(splitAssessorMock);
      reset(splitAssessorFactoryMock);
      reset(splitterMock);
      reset(splitterFactoryMock);
    });

    test('should create a GreedyTreeSplitSelector instance', () {
      final type = TreeSplitSelectorType.greedy;
      final assessorType = TreeSplitAssessorType.majority;
      final splitterType = TreeSplitterType.greedy;

      final splitSelector =
          factory.createByType(type, assessorType, splitterType);

      expect(splitSelector, isA<GreedyTreeSplitSelector>());
    });

    test(
        'should call split assessor factory while creating the '
        'instance', () {
      final type = TreeSplitSelectorType.greedy;
      final assessorType = TreeSplitAssessorType.majority;
      final splitterType = TreeSplitterType.greedy;

      factory.createByType(type, assessorType, splitterType);

      verify(splitAssessorFactoryMock.createByType(assessorType)).called(1);
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
