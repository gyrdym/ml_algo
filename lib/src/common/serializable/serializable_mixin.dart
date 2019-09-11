import 'dart:convert';
import 'dart:io';

import 'package:ml_algo/src/common/serializable/serializable.dart';

mixin SerializableMixin implements Serializable {
  @override
  Future<File> saveAsJSON(String fileName) async {
    final file = await File(fileName).create(recursive: true);
    final json = jsonEncode(serialize());
    return file.writeAsString(json);
  }
}
