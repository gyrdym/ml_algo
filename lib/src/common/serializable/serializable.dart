import 'dart:io';

abstract class Serializable {
  Map<String, dynamic> serialize();
  Future<File> saveAsJSON(String fileName);
}
