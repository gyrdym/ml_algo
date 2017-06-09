import 'cross_validator.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/data_splitter/interface/leave_p_out_splitter.dart';

class LpoCrossValidator extends CrossValidator {
  LpoCrossValidator({int p = 5}) :
        super(injector.get(LeavePOutSplitter)..configure(p: p));
}
