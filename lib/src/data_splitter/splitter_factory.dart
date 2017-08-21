part of 'package:dart_ml/src/dart_ml_impl.dart';

class DataSplitterFactory {
  static KFoldSplitter KFold() => new _KFoldSplitterImpl();
  static LeavePOutSplitter Lpo() => new _LeavePOutSplitterImpl();
}