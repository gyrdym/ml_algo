import 'package:dart_ml/src/data_splitter/interface/splitter.dart';

abstract class LeavePOutSplitter implements Splitter {
  void configure({int p});
}