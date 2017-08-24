part of 'package:dart_ml/src/interface.dart';

abstract class LeavePOutSplitter implements Splitter {
  void configure({int p});
}