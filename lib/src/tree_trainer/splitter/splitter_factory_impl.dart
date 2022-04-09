import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';

class TreeSplitterFactoryImpl implements TreeSplitterFactory {
  TreeSplitterFactoryImpl(
    this._assessorFactory,
    this._nominalSplitterFactory,
    this._numericalSplitterFactory,
  );

  final TreeSplitAssessorFactory _assessorFactory;
  final NominalTreeSplitterFactory _nominalSplitterFactory;
  final NumericalTreeSplitterFactory _numericalSplitterFactory;

  @override
  TreeSplitter createByType(
      TreeSplitterType type, TreeAssessorType assessorType) {
    final assessor = _assessorFactory.createByType(assessorType);
    final numericalSplitter = _numericalSplitterFactory.create();
    final nominalSplitter = _nominalSplitterFactory.create();

    switch (type) {
      case TreeSplitterType.greedy:
        return GreedyTreeSplitter(assessor, numericalSplitter, nominalSplitter);

      default:
        throw UnsupportedError('Tree splitter type $type is not '
            'supported');
    }
  }
}
