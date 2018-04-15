import 'package:dart_ml/src/core/data_splitter/k_fold.dart';
import 'package:dart_ml/src/core/data_splitter/leave_p_out.dart';
import 'package:dart_ml/src/core/data_splitter/splitter.dart';
import 'package:dart_ml/src/core/data_splitter/type.dart';

class DataSplitterFactory {
  static Splitter KFold(int numberOfFolds) => new KFoldSplitterImpl(numberOfFolds);

  static Splitter LPO(int p) => new LeavePOutSplitterImpl(p);

  static Splitter createByType(SplitterType type, int value) {
    Splitter splitter;

    switch(type) {
      case SplitterType.KFOLD:
        splitter = KFold(value);
        break;

      case SplitterType.LPO:
        splitter = LPO(value);
        break;

      default:
        throw new UnsupportedError('Unsupported splitter type: ${type}');
    }

    return splitter;
  }
}