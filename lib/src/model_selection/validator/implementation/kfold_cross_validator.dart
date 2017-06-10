import 'package:dart_ml/src/model_selection/validator/implementation/base_cross_validator.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/data_splitter/interface/k_fold_splitter.dart';
import 'package:dart_ml/src/model_selection/validator/interface/kfold_cross_validator.dart';

class KFoldCrossValidatorImpl extends BaseCrossValidator implements KFoldCrossValidator {
  KFoldCrossValidatorImpl({int numberOfFolds = 5}) :
        super(injector.get(KFoldSplitter)..configure(numberOfFolds: numberOfFolds));
}
