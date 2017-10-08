part of 'package:dart_ml/src/core/implementation.dart';

class DataSplitterFactory {
  static Splitter KFold(int numberOfFolds) => new _KFoldSplitterImpl(numberOfFolds);

  static Splitter LPO(int p) => new _LeavePOutSplitterImpl(p);

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