import 'cross_validator.dart';
import 'package:dart_ml/src/data_splitter/k_fold_splitter.dart';

class KFoldCrossValidator extends CrossValidator {
  KFoldCrossValidator({int numberOfFolds = 5}) : super(new KFoldSplitter(numberOfFolds: numberOfFolds));
}
