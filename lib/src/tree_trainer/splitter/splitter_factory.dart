import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';

abstract class TreeSplitterFactory {
  TreeSplitter createByType(
    TreeSplitterType type,
    TreeAssessorType assessorType,
  );
}
