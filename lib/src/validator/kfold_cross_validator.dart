import 'cross_validator.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/data_splitter/interface/k_fold_splitter.dart';

class KFoldCrossValidator extends CrossValidator {
  KFoldCrossValidator({int numberOfFolds = 5}) :
        super(injector.get(KFoldSplitter)..configure(numberOfFolds: numberOfFolds));
}
