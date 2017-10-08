part of 'package:dart_ml/src/core/implementation.dart';

class DataSplitterFactory {
  static Splitter KFold() => new _KFoldSplitterImpl();

  static Splitter LPO() => new _LeavePOutSplitterImpl();

  static Splitter createByType(SplitterType type) {
    Splitter splitter;

    switch(type) {
      case SplitterType.KFOLD:
        splitter = KFold();
        break;

      case SplitterType.LPO:
        splitter = LPO();
        break;

      default:
        throw new UnsupportedError('Unsupported splitter type: ${type}');
    }

    return splitter;
  }
}