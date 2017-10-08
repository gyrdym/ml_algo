part of 'package:dart_ml/src/core/interface.dart';

abstract class LeavePOutSplitter implements Splitter {
  void configure({int p});
}