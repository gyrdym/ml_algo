part of 'package:dart_ml/src/dart_ml.dart';

abstract class LeavePOutSplitter implements Splitter {
  void configure({int p});
}