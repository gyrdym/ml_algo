import 'package:dart_ml/src/data_splitter/base.dart';

abstract class LeavePOutSplitter implements Splitter {
  void configure({int p});
}