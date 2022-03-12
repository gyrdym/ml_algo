import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:test/test.dart';

import '../../../mocks.mocks.dart';

void main() {
  group('NumericalTreeSplitterFactoryImpl', () {
    final nodeFactoryMock = MockIntermediateTreeNodeFactory();
    final factory = NumericalTreeSplitterFactoryImpl(nodeFactoryMock);

    test('should create a NominalTreeSplitterImpl instance', () {
      final splitter = factory.create();

      expect(splitter, isA<NumericalTreeSplitterImpl>());
    });
  });
}
