import 'package:ml_algo/src/tree_solver/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_solver/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_solver/splitter/numerical_splitter/numerical_splitter_impl.dart';

class NumericalTreeSplitterFactoryImpl implements
    NumericalTreeSplitterFactory {

  const NumericalTreeSplitterFactoryImpl();

  @override
  NumericalTreeSplitter create() =>
      const NumericalTreeSplitterImpl();
}
