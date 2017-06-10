import 'package:dart_ml/src/model_selection/validator/implementation/base_cross_validator.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/data_splitter/interface/leave_p_out_splitter.dart';
import 'package:dart_ml/src/model_selection/validator/interface/lpo_cross_validator.dart';

class LpoCrossValidatorImpl extends BaseCrossValidator implements LpoCrossValidator {
  LpoCrossValidatorImpl({int p = 5}) :
        super(injector.get(LeavePOutSplitter)..configure(p: p));
}
