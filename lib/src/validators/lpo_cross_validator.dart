import 'cross_validator.dart';
import 'package:dart_ml/src/data_splitters/leave_p_out_splitter.dart';

class LpoCrossValidator extends CrossValidator {
  LpoCrossValidator({int p = 5}) : super(new LeavePOutSplitter(p: p));
}
