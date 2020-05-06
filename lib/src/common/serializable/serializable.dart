import 'dart:io';

abstract class Serializable {
  /// Returns a serialized object
  Map<String, dynamic> toJson();

  /// Saves a json file in [fileName] file
  Future<File> saveAsJson(String fileName);
}
