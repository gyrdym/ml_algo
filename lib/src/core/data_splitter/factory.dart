part of 'package:dart_ml/src/core/implementation.dart';

class DataSplitterFactory {
  static KFoldSplitter createKFoldSplitter() => new _KFoldSplitterImpl();
  static LeavePOutSplitter createLpoSplitter() => new _LeavePOutSplitterImpl();
}