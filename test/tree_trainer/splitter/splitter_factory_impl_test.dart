import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('TreeSplitterFactoryImpl', () {
    final assessorMock = TreeSplitAssessorMock();
    final assessorFactoryMock = createTreeSplitAssessorFactoryMock(assessorMock);

    final nominalSplitterMock = NominalTreeSplitterMock();
    final nominalSplitterFactoryMock = createNominalTreeSplitterFactoryMock(
        nominalSplitterMock);

    final numericalSplitterMock = NumericalTreeSplitterMock();
    final numericalSplitterFactoryMock = createNumericalTreeSplitterFactoryMock(
        numericalSplitterMock);

    final factory = TreeSplitterFactoryImpl(assessorFactoryMock,
        nominalSplitterFactoryMock, numericalSplitterFactoryMock);

    tearDown(() {
      reset(assessorMock);
      reset(assessorFactoryMock);
      reset(nominalSplitterMock);
      reset(nominalSplitterFactoryMock);
      reset(numericalSplitterMock);
      reset(numericalSplitterFactoryMock);
    });

    test('should create a GreedyTreeSplitter instance', () {
      final type = TreeSplitterType.greedy;
      final assessorType = TreeSplitAssessorType.majority;

      final splitter = factory.createByType(type, assessorType);

      expect(splitter, isA<GreedyTreeSplitter>());
    });

    test('should call assessor factory while creating the instance', () {
      final type = TreeSplitterType.greedy;
      final assessorType = TreeSplitAssessorType.majority;

      factory.createByType(type, assessorType);

      verify(assessorFactoryMock.createByType(assessorType)).called(1);
    });

    test('should call nominal splitter factory while creating the '
        'instance', () {
      final type = TreeSplitterType.greedy;
      final assessorType = TreeSplitAssessorType.majority;

      factory.createByType(type, assessorType);

      verify(nominalSplitterFactoryMock.create()).called(1);
    });

    test('should call numerical splitter factory while creating the '
        'instance', () {
      final type = TreeSplitterType.greedy;
      final assessorType = TreeSplitAssessorType.majority;

      factory.createByType(type, assessorType);

      verify(numericalSplitterFactoryMock.create()).called(1);
    });
  });
}
