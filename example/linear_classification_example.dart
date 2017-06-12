import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

Future main() async {
  Dependencies.configure();

  csv.CsvCodec csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/datasets/LSVT_voice_rehabilitation.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) =>
      item.map((Object feature) {
        if (feature is String) {
          return double.parse(feature.replaceFirst(new RegExp('\,'), '.'));
        }

        return (feature as num).toDouble();
      }).toList();

  List<Vector> features = fields
      .map((List item) => new Vector.from(extractFeatures(item.sublist(0, item.length - 1))))
      .toList(growable: false);

  Vector labels = new Vector.from(fields.map((List<num> item) => item.last.toDouble()).toList(growable: false));
}