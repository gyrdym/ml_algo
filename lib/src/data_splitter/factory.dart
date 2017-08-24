part of 'package:dart_ml/src/dart_ml_impl.dart';

class DataSplitterFactory {
  static KFoldSplitter createKFoldSplitter() => new _KFoldSplitterImpl();
  static LeavePOutSplitter createLpoSplitter() => new _LeavePOutSplitterImpl();
}