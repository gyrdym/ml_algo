import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_impl.dart';
import 'package:test/test.dart';

void main() {
  group('NominalTreeSplitterFactoryImpl', () {
    final factory = const NominalTreeSplitterFactoryImpl();

    test('should create a NominalTreeSplitterImpl instance', () {
      final splitter = factory.create();

      expect(splitter, isA<NominalTreeSplitterImpl>());
    });
  });
}
