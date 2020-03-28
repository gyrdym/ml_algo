import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:test/test.dart';

void main() {
  group('NumericalTreeSplitterFactoryImpl', () {
    final factory = const NumericalTreeSplitterFactoryImpl();

    test('should create a NominalTreeSplitterImpl instance', () {
      final splitter = factory.create();

      expect(splitter, isA<NumericalTreeSplitterImpl>());
    });
  });
}
